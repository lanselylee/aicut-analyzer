"""Speech: what a clip says and when, the pauses worth cutting, subtitle lines, and how far to lower music under it.

Everything after transcription works on a list of timed words, so it is tested without a model:
- `Transcriber` turns a file into words (`WhisperTranscriber` runs Whisper locally on Apple silicon; a hosted service
  can be another transcriber). `CachedTranscriber` keeps each file's words so it is transcribed once.
- `keep_ranges` is what a jump cut keeps: the speech, each pause longer than `max_pause` cut down, fillers ("嗯",
  "呃"), the first try of a repeated word or restarted phrase (`repeated`) and earlier takes of a sentence said
  again (`retakes`) cut out, never inside a word.
  `snap_to_quiet` then moves each cut to the quietest moment near it.
- `subtitle_lines` groups words into lines a viewer reads at once; `to_vtt` writes them as WebVTT with each word timed
  (what OpenFilm Studio makes subtitles from).
- `duck_gain` is the music's gain over time: lowered while someone speaks, eased down just before and back up after.

Whisper writes words where nobody speaks: over music or wind it "hears" a stock phrase from its training subtitles
("字幕志愿者 李宗盛", "Thank you for watching"), and it can loop on one phrase. `WhisperTranscriber` drops such segments.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    start: float    # seconds in the file
    end: float
    prob: float = 1.0


@dataclass(frozen=True)
class Line:
    words: tuple[Word, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return join_words([w.text for w in self.words])


CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
SENTENCE_END = re.compile(r"[。！？!?.…]$")
CLAUSE_END = re.compile(r"[，、；：,;:]$")


def join_words(parts: list[str]) -> str:
    """Words as they are written: a space between words, none next to CJK."""
    out = ""
    for p in (p.strip() for p in parts):
        if not p:
            continue
        out += " " + p if out and not CJK.search(out[-1]) and not CJK.search(p[0]) else p
    return out


# ---------------------------------------------------------------- transcription

FULL_WIDTH = str.maketrans({",": "，", "?": "？", "!": "！", ":": "：", ";": "；"})


def cjk_punct(text: str) -> str:
    """ASCII punctuation in Chinese made full-width ("好," → "好，"), so it reads right and needs no space after."""
    return text.translate(FULL_WIDTH) if CJK.search(text) else text


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, path: Path) -> list[Word]:
        """The words said in a media file, in its own seconds."""


HALLUCINATIONS = re.compile(r"字幕|订阅|點贊|点赞|謝謝觀看|谢谢观看|明镜|李宗盛|作词|作詞|作曲|编曲|編曲|初音|ミク|"
                            r"thanks?( you)? for watching|subscribe|amara", re.I)
ONE_SOUND = re.compile(r"^(\w)\1{2,}$")       # "喇喇喇喇": a sizzle or a hum heard as a syllable


def plausible(segment: dict) -> bool:
    """Whether a Whisper segment is speech rather than something it made up over noise or music."""
    text = segment.get("text", "").strip()
    if not text or HALLUCINATIONS.search(text):
        return False
    if segment.get("compression_ratio", 0) > 2.4:          # the same few words over and over
        return False
    if segment.get("avg_logprob", 0) < -1.2:               # guessed: over cooking or music it is -1.6 to -4.7
        return False
    if ONE_SOUND.match(re.sub(r"[\s" + re.escape(PUNCT) + "]", "", text)):
        return False
    if segment.get("no_speech_prob", 0) > 0.6 and segment.get("avg_logprob", 0) < -0.8:
        return False
    words = segment.get("words") or []
    return not words or np.mean([w.get("probability", 1) for w in words]) >= 0.3


class WhisperTranscriber(Transcriber):
    """Whisper on Apple silicon (mlx-whisper, the `speech` extra), with word times. `language` None detects it; "zh"
    also asks for simplified characters. (A prompt with a full-width comma in it makes the turbo model write "Ｂ" for
    every comma.) Commas and stops after Chinese are made full-width, and with "zh", traditional characters, which
    it still writes for some stretches, simplified (OpenCC)."""

    PROMPTS = {"zh": "以下是简体中文普通话的句子。"}

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo", language: str | None = None):
        self.model = model
        self.language = language
        self.prompt = self.PROMPTS.get(language or "")
        self.version = 3          # what it writes changed: cached words from an older one are not reused

    def transcribe(self, path: Path) -> list[Word]:
        import mlx_whisper

        out = mlx_whisper.transcribe(str(path), path_or_hf_repo=self.model, language=self.language,
                                     word_timestamps=True, condition_on_previous_text=False,
                                     initial_prompt=self.prompt)
        fix = cjk_punct
        if self.language == "zh":
            from opencc import OpenCC
            t2s = OpenCC("t2s").convert
            fix = lambda text: cjk_punct(t2s(text))  # noqa: E731
        return [Word(fix(w["word"].strip()), round(w["start"], 3), round(w["end"], 3),
                     round(w.get("probability", 1), 3))
                for seg in out["segments"] if plausible(seg) for w in seg.get("words", [])
                if w["word"].strip(PUNCT + "・·")]


UNIT = re.compile(r"[A-Za-z0-9']+|[぀-ヿ㐀-鿿가-힯]")


def sentence_words(sentence: dict) -> list[Word]:
    """FunASR's sentence ({"text", "timestamp": [[ms, ms], ...]}) as words: a Chinese character or a Latin word per
    timestamp, with the punctuation after it. When the count of units and timestamps differ, the sentence's span is
    shared evenly between its units."""
    text = sentence["text"]
    units: list[str] = []
    for m in re.finditer(r"([A-Za-z0-9']+|[぀-ヿ㐀-鿿가-힯])([^A-Za-z0-9'぀-ヿ㐀-鿿가-힯]*)", text):
        units.append(m.group(1) + m.group(2).strip())
    stamps = sentence.get("timestamp") or []
    if len(stamps) != len(units):
        a, b = sentence["start"], sentence["end"]
        step = (b - a) / max(len(units), 1)
        stamps = [[a + i * step, a + (i + 1) * step] for i in range(len(units))]
    return [Word(cjk_punct(u), round(a / 1000, 3), round(b / 1000, 3)) for u, (a, b) in zip(units, stamps)]


class ParaformerTranscriber(Transcriber):
    """Paraformer (FunASR, the `asr` extra), a model made for Mandarin: on a kitchen vlog it heard "花纹，但是被我洗掉了"
    and "今日便利店" where Whisper wrote "花门单都洗掉了" and "机日便利店", and it writes the fillers ("嗯") Whisper leaves
    out, so they can be cut. A voice-activity model finds the speech and a punctuation model punctuates it. Its
    models (about 1 GB) are downloaded once to ~/.cache/modelscope."""

    def __init__(self, model: str = "paraformer-zh"):
        self.model = model
        self.language = "zh"
        self.version = 1
        self._asr = None

    def transcribe(self, path: Path) -> list[Word]:
        if self._asr is None:
            from funasr import AutoModel
            self._asr = AutoModel(model=self.model, vad_model="fsmn-vad", punc_model="ct-punc",
                                  disable_update=True, log_level="ERROR")
        out = self._asr.generate(input=str(path), batch_size_s=300, sentence_timestamp=True)
        if not out:     # the voice detector found no speech at all (a garage, a street): nothing was said
            return []
        return [w for s in (out[0].get("sentence_info") or []) for w in sentence_words(s)]


class CachedTranscriber(Transcriber):
    """Keeps each file's words as JSON in `folder`, keyed by the file's path, size and modification time: a file is
    transcribed again only when it changes or the transcriber does."""

    def __init__(self, inner: Transcriber, folder: Path):
        self.inner = inner
        self.folder = folder

    def key(self, path: Path) -> str:
        st = path.stat()
        who = ":".join(str(getattr(self.inner, k, "")) for k in ("model", "language", "prompt", "version"))
        who = f"{type(self.inner).__name__}:{who}"
        return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{who}".encode()).hexdigest()[:16]

    def transcribe(self, path: Path) -> list[Word]:
        cache = self.folder / f"{self.key(path)}.json"
        if cache.exists():
            return [Word(**w) for w in json.loads(cache.read_text())]
        words = self.inner.transcribe(path)
        self.folder.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([asdict(w) for w in words], ensure_ascii=False))
        return words


# ---------------------------------------------------------------- what to keep

FILLERS = {"嗯", "恩", "呃", "额", "唔", "噢", "um", "uh", "erm", "hmm", "mm"}
PUNCT = " ,.，。、!?！？…:：;；\"'“”"


def bare(text: str) -> str:
    return text.strip(PUNCT).lower()


def is_filler(w: Word) -> bool:
    return bare(w.text) in FILLERS


def repeated(words: list[Word], max_n: int = 4, max_gap: float = 1.5, stutter_gap: float = 0.12) -> set[int]:
    """Indices of words said again right after: a stutter ("我…我们") or a restart ("我们今天 我们今天去"). The first
    try is dropped, the one the speaker went on with is kept. Fillers between the tries are looked through. A single
    character said twice counts only with a pause between (`stutter_gap`): "谢谢", "看看" in one breath are words. The
    last word of a try may be cut short ("我们今" before "我们今天"), so it need only start the word said again."""
    idx = [i for i, w in enumerate(words) if not is_filler(w) and bare(w.text)]
    texts = [bare(words[i].text) for i in idx]
    drop: set[int] = set()
    k = 0
    while k < len(idx):
        for n in range(min(max_n, (len(idx) - k) // 2), 0, -1):
            a, b = texts[k:k + n], texts[k + n:k + 2 * n]
            if a[:-1] != b[:-1] or not b[-1].startswith(a[-1]):
                continue
            gap = words[idx[k + n]].start - words[idx[k + n - 1]].end
            if gap <= max_gap and (n > 1 or gap >= stutter_gap or (CJK.search(b[0]) and len(a[0]) > 1)):
                drop.update(idx[k:k + n])
                k += n
                break
        else:
            k += 1
    return drop


def sentences(words: list[Word], pause: float = 0.6) -> list[list[int]]:
    """Indices of words grouped into sentences: ended by a stop ("。", "?") or a pause longer than `pause`."""
    out: list[list[int]] = []
    for i, w in enumerate(words):
        if not out or SENTENCE_END.search(words[i - 1].text) or w.start - words[i - 1].end > pause:
            out.append([])
        out[-1].append(i)
    return out


def retake_of(take: str, later: str, alike: float = 0.75, slack: float = 0.4, ending: int = 2) -> bool:
    """Whether `later` starts by saying `take` again: some prefix of it, within `slack` of the take's length (a
    retake often leaves words out: "因为", "我们"), is at least `alike` the same (difflib's ratio) and ends with the same `ending` characters. What the
    speaker goes on to say after the retake does not count against it. The ending is what tells a retake ("因为今天
    我们是一天的课" … "今天是一天的课") from two lines built alike ("我很喜欢这个。我很喜欢那个"), which differ at
    the end."""
    d = max(2, round(len(take) * slack))
    for n in range(max(ending, len(take) - d), min(len(later), len(take) + d) + 1):
        head = later[:n]
        if head[-ending:] == take[-ending:] and SequenceMatcher(None, take, head).ratio() >= alike:
            return True
    return False


def retakes(words: list[Word], window: float = 12.0, alike: float = 0.75, shortest: int = 3, max_run: int = 3,
            max_between: int = 12) -> set[int]:
    """Indices of words a speaker said and then said again: the earlier take, and whatever they began between the
    takes and gave up on ("因为今天我们是一天的课 / 但是我们去 / 今天是一天的课，他中午没吃饭" keeps only the last).
    A take (up to `max_run` sentences, for one that paused in the middle, at least `shortest` characters) is said
    again when a sentence starting within `window` seconds of its end begins with it (`retake_of`). What lies
    between is dropped with it only when it is at most `max_between` characters: more than that, and the speaker
    went on to something else, so it is no retake."""
    sents = [[i for i in s if not is_filler(words[i])] for s in sentences(words)]
    sents = [s for s in sents if s]
    text = ["".join(re.sub(r"[\s" + re.escape(PUNCT) + "]", "", bare(words[i].text)) for i in s) for s in sents]
    drop: set[int] = set()
    for a in range(len(sents)):
        for e in range(a, min(a + max_run, len(sents))):      # a take that paused in the middle is sentences a..e
            take = "".join(text[a:e + 1])
            if len(take) < shortest:
                continue
            end = words[sents[e][-1]].end
            again = None
            for b in range(e + 1, len(sents)):
                if words[sents[b][0]].start - end > window or len("".join(text[e + 1:b])) > max_between:
                    break
                if retake_of(take, "".join(text[b:b + max_run]), alike):
                    again = b
                    break
            if again is not None:
                drop.update(i for k in range(a, again) for i in sents[k])
                break
    return drop


@dataclass(frozen=True)
class Removed:
    start: float
    end: float
    why: str        # "pause", "filler", "repeat" or "retake"
    text: str = ""


def within(words: list[Word], lo: float, hi: float) -> list[Word]:
    return [w for w in words if w.start >= lo - 1e-6 and w.end <= hi + 1e-6]


def speech_share(words: list[Word], lo: float, hi: float) -> float:
    """How much of [lo, hi] is someone speaking (0 to 1)."""
    return sum(w.end - w.start for w in within(words, lo, hi)) / max(hi - lo, 1e-9)


def dropped(words: list[Word], drop_fillers: bool = True, drop_repeats: bool = True) -> dict[int, str]:
    """{index: why} for the words a jump cut leaves out."""
    out = {i: "filler" for i, w in enumerate(words) if drop_fillers and is_filler(w)}
    if drop_repeats:
        out.update({i: "repeat" for i in repeated(words)})
        out.update({i: "retake" for i in retakes(words)})
    return out


def keep_ranges(words: list[Word], lo: float, hi: float, max_pause: float = 0.4, lead: float = 0.12,
                tail: float = 0.18, drop_fillers: bool = True, drop_repeats: bool = True) -> list[tuple[float, float]]:
    """The stretches of [lo, hi] a jump cut keeps: the words, with a pause longer than `max_pause` cut so `tail`
    of it stays after the last word and `lead` before the next, and fillers and first tries of repeated words cut
    out. Edges are always between words, never inside one. Without words, all of [lo, hi]."""
    said = within(words, lo, hi)
    if not said:
        return [(lo, hi)]
    out_of = dropped(said, drop_fillers, drop_repeats)
    keep = [i for i in range(len(said)) if i not in out_of]
    if not keep:
        return []
    first, last = said[keep[0]], said[keep[-1]]
    before = [said[i] for i in range(keep[0])]
    out = [[max(lo, first.start - lead, *(w.end for w in before)), first.end]]
    for p, q in zip(keep, keep[1:]):
        prev, w = said[p], said[q]
        between = said[p + 1:q]
        end, start = prev.end + tail, w.start - lead
        if between:
            end, start = min(end, between[0].start), max(start, between[-1].end)
        if between or (w.start - prev.end > max_pause and end < start):
            out[-1][1] = end
            out.append([start, w.end])
        else:
            out[-1][1] = w.end
    after = said[keep[-1] + 1:]
    out[-1][1] = min(hi, last.end + tail, *(w.start for w in after))
    return [(round(a, 3), round(b, 3)) for a, b in out]


def removed(words: list[Word], lo: float, hi: float, ranges: list[tuple[float, float]], **kw) -> list[Removed]:
    """What the jump cut takes out of [lo, hi], each gap between kept ranges named by the words in it."""
    said = within(words, lo, hi)
    out_of = dropped(said, kw.get("drop_fillers", True), kw.get("drop_repeats", True))
    edges = [lo, *(t for r in ranges for t in r), hi]
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 1e-3:
            continue
        gone = [i for i, w in enumerate(said) if w.start >= a - 1e-6 and w.end <= b + 1e-6]
        whys = {out_of.get(i, "pause") for i in gone}
        why = next((k for k in ("retake", "repeat", "filler") if k in whys), "pause")
        out.append(Removed(round(a, 3), round(b, 3), why, join_words([said[i].text for i in gone])))
    return out


def tighten(words: list[Word], loud: np.ndarray, rate: float, floor: float = 0.1,
            shortest: float = 0.08) -> list[Word]:
    """Words with the silence at their edges taken off. Whisper stretches the first word after a pause back into the
    pause (a 0.2 s word marked 0.6 s), so a cut there would keep the silence. A word's edge moves in to where the
    sound first rises above `floor` of the speech's level (the 90th percentile inside words), and never leaves the word
    shorter than `shortest`."""
    if not words or not len(loud):
        return list(words)
    inside = np.concatenate([loud[int(w.start * rate):int(w.end * rate)] for w in words] or [loud])
    thr = floor * float(np.percentile(inside, 90)) if len(inside) else 0.0
    out = []
    for w in words:
        a, b = int(w.start * rate), min(len(loud), int(np.ceil(w.end * rate)))
        on = np.nonzero(loud[a:b] > thr)[0]
        if len(on) == 0:
            out.append(w)
            continue
        start = max(w.start, (a + on[0]) / rate)
        end = min(w.end, (a + on[-1] + 1) / rate)
        if end - start < shortest:
            mid = (start + end) / 2
            start, end = max(w.start, mid - shortest / 2), min(w.end, mid + shortest / 2)
        out.append(Word(w.text, round(start, 3), round(end, 3), w.prob))
    return out


def snap_to_quiet(ranges: list[tuple[float, float]], words: list[Word], loud: np.ndarray, rate: float,
                  reach: float = 0.08) -> list[tuple[float, float]]:
    """Each cut edge moved, by at most `reach` seconds, to the quietest moment near it (`loud` is the sound's level
    sampled `rate` times a second), so a cut lands in silence rather than on a breath or the tail of a word. An edge
    never moves into a word, nor past its neighbour."""
    inside = np.zeros(len(loud), bool)
    for w in words:
        inside[max(0, int(np.ceil(w.start * rate))):int(w.end * rate) + 1] = True
    flat = [t for r in ranges for t in r]
    out = []
    for i, t in enumerate(flat):
        lo = max(t - reach, out[-1] if out else 0)
        hi = min(t + reach, flat[i + 1] if i + 1 < len(flat) else t + reach)
        cand = [j for j in range(int(np.ceil(lo * rate)), int(hi * rate) + 1) if 0 <= j < len(loud) and not inside[j]]
        if not cand:
            out.append(t)
            continue
        quiet = min(loud[j] for j in cand)
        best = min((j for j in cand if loud[j] <= quiet * 1.05 + 1e-9), key=lambda j: abs(j / rate - t))
        out.append(round(best / rate, 3))
    return [(a, b) for a, b in zip(out[::2], out[1::2])]


def loudness(samples: np.ndarray, sr: int, rate: int = 100) -> np.ndarray:
    """RMS level of (n, channels) samples, `rate` values a second."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    hop = sr // rate
    n = len(mono) // hop
    return np.sqrt((mono[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


# ---------------------------------------------------------------- subtitles

def subtitle_lines(words: list[Word], gap: float = 0.45, max_chars: int = 18, max_seconds: float = 5.0,
                   drop_fillers: bool = True) -> list[Line]:
    """Words grouped into lines: a new line after a pause longer than `gap`, after a sentence ends, and before a line
    gets longer than `max_chars` (CJK) or 2.3 × that (other scripts) or `max_seconds`, preferring to break after a
    comma."""
    said = [w for w in words if not (drop_fillers and is_filler(w))]
    lines: list[list[Word]] = []
    for w in said:
        cur = lines[-1] if lines else None
        if cur is not None:
            text = join_words([x.text for x in cur + [w]])
            limit = max_chars if CJK.search(text) else int(max_chars * 2.3)
            if (w.start - cur[-1].end > gap or SENTENCE_END.search(cur[-1].text)
                    or len(text) > limit or w.end - cur[0].start > max_seconds):
                if not (w.start - cur[-1].end > gap or SENTENCE_END.search(cur[-1].text)):
                    # too long: break at the last comma in it when there is one
                    k = max((i for i, x in enumerate(cur[:-1]) if CLAUSE_END.search(x.text)), default=None)
                    if k is not None:
                        lines[-1], rest = cur[:k + 1], cur[k + 1:]
                        lines.append(rest + [w])
                        continue
                cur = None
        if cur is None:
            lines.append([w])
        else:
            cur.append(w)
    return [Line(tuple(ws)) for ws in lines]


def _stamp(t: float) -> str:
    ms = max(0, round(t * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(lines: list[Line], note: str = "") -> str:
    """WebVTT: a cue per line in the file's seconds, a timestamp tag before each word after the first."""
    cues = []
    for line in lines:
        text = ""
        for i, w in enumerate(line.words):
            piece = join_words([text, w.text])[len(text):]       # the word with the space it needs
            if i:
                lead = len(piece) - len(piece.lstrip())
                piece = piece[:lead] + f"<{_stamp(w.start)}>" + _esc(piece[lead:])
            else:
                piece = _esc(piece)
            text += piece
        cues.append(f"{_stamp(line.start)} --> {_stamp(line.end)}\n{text}")
    head = f"WEBVTT\n\n{f'NOTE {note}' + chr(10) * 2 if note else ''}"
    return head + "\n\n".join(cues) + "\n"


# ---------------------------------------------------------------- music under speech

def merge_spans(spans: list[tuple[float, float]], bridge: float) -> list[tuple[float, float]]:
    """Spans sorted and joined where the gap between them is at most `bridge`."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= bridge:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def duck_gain(t: np.ndarray, speech: list[tuple[float, float]], level: float = 0.25, attack: float = 0.25,
              release: float = 0.6, bridge: float = 1.2) -> np.ndarray:
    """Music gain at times `t`: `level` while someone speaks, 1 elsewhere. It starts down `attack` seconds before
    speech so the first word is clear, comes back up over `release` after it, and stays down across pauses shorter
    than `bridge` so it does not pump between words. Ramps are smoothstep, not linear, which ears hear as even."""
    t = np.asarray(t, np.float64)
    duck = np.zeros_like(t)
    for a, b in merge_spans(speech, bridge):
        down = np.clip((t - (a - attack)) / attack, 0, 1)
        up = np.clip(1 - (t - b) / release, 0, 1)
        d = np.minimum(down, up)
        duck = np.maximum(duck, d * d * (3 - 2 * d))
    return 1 - (1 - level) * duck


def read_audio(path: Path, rate: int = 48000) -> np.ndarray:
    """A file's sound as float32 samples (n, 2) at `rate`, through ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(rate),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def write_wav(path: Path, samples: np.ndarray, rate: int = 48000) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ac", str(samples.shape[1]), "-ar", str(rate),
                    "-i", "-", str(path)], input=pcm.tobytes(), check=True)
