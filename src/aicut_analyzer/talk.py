"""aicut-talk <video or folder> ... [--music bed.mp3] [--out folder] [--language zh] [--max-pause 0.4]

Cuts talking-head recordings into an OpenFilm project: the pauses, fillers ("嗯", "呃"), first tries of repeated
words and earlier takes of a sentence said again cut out (a clip per kept stretch of the same file), subtitles as a
WebVTT beside each video (Studio draws and burns them), and with --music, the music lowered under the speech, written
as a treated file on its own track. Prints what it cut, and why, so a cut can be checked before it is trusted.

Several videos (or a folder) are put in the order they were shot, by the time the camera wrote into each file
(an iPhone's com.apple.quicktime.creationdate, with its time zone), not by name or file date, which copying changes.
Short clips (under --short seconds) are kept whole, as b-roll; a gap of more than --scene-gap minutes starts a scene.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np

from .speech import (CachedTranscriber, Removed, Transcriber, Word, duck_gain, keep_ranges, loudness, read_audio,
                     removed, snap_to_quiet, subtitle_lines, tighten, to_vtt, within, write_wav)

RATE = 48000
CACHE = Path.home() / ".cache" / "aicut" / "words"


@dataclass(frozen=True)
class Probe:
    duration: float
    width: int      # as it is shown: a phone's portrait clip, stored landscape with a rotation, is tall
    height: int
    shot_at: datetime | None = None     # when recording began, as the camera wrote it


def shot_time(tags: dict) -> datetime | None:
    """When a recording began: an iPhone's own creation date (local time with its offset), else the container's
    creation_time (UTC). None when the file says neither (a screen recording, an export)."""
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        if key in tags:
            try:
                return datetime.fromisoformat(tags[key].replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def probe(path: Path) -> Probe:
    out = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height:stream_side_data=rotation:format=duration:format_tags",
                                     "-of", "json", str(path)], capture_output=True, check=True, text=True).stdout)
    s = out["streams"][0]
    rot = next((int(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d), 0)
    w, h = (s["height"], s["width"]) if rot % 180 else (s["width"], s["height"])
    # to the millisecond, down: a clip's #t= may not end past the file, and film times have three decimals
    duration = math.floor(float(out["format"]["duration"]) * 1000) / 1000
    return Probe(duration, w, h, shot_time(out["format"].get("tags", {})))


@dataclass(frozen=True)
class Shot:
    path: Path
    probe: Probe


def in_shooting_order(shots: list[Shot]) -> list[Shot]:
    """Shots by when they were shot; those without a time after the rest, by name."""
    return sorted(shots, key=lambda s: (s.probe.shot_at is None,
                                        s.probe.shot_at.timestamp() if s.probe.shot_at else 0, s.path.name))


def local(when: datetime) -> datetime:
    """A shot's time on the clock where it was taken. An iPhone writes its own time zone, which is kept; a time in
    UTC (or with no zone) only says the instant, so it is put in this computer's time zone."""
    off = when.utcoffset()
    return when if off is not None and off.total_seconds() != 0 else when.astimezone()


def day(shot: Shot) -> str:
    """The day a shot was taken, as MM-DD where it was taken ("" without a time)."""
    return local(shot.probe.shot_at).strftime("%m-%d") if shot.probe.shot_at else ""


def in_day_order(shots: list[Shot], days: list[str]) -> list[Shot]:
    """Shots in shooting order within each day, the days in the order the person gives (MM-DD, or YYYY-MM-DD);
    days not given follow, in shooting order. A film need not tell its days in the order they happened."""
    rank = {d[-5:]: i for i, d in enumerate(days)}
    ordered = in_shooting_order(shots)
    return sorted(ordered, key=lambda s: rank.get(day(s), len(rank)))


def scenes(shots: list[Shot], gap: float = 600.0) -> list[list[Shot]]:
    """Shots in order, split where more than `gap` seconds pass between one ending and the next beginning, or
    where the next was shot earlier (the film goes back to another day)."""
    out: list[list[Shot]] = []
    for s in shots:
        prev = out[-1][-1] if out else None
        if (prev is None or s.probe.shot_at is None or prev.probe.shot_at is None
                or not 0 <= (s.probe.shot_at - prev.probe.shot_at).total_seconds() - prev.probe.duration <= gap):
            out.append([])
        out[-1].append(s)
    return out


@dataclass(frozen=True)
class TalkCut:
    """A jump cut of one recording: what it keeps (in the file's seconds, each stretch at its speed) and what it
    takes out."""
    ranges: list[tuple[float, float]]
    removed: list[Removed]
    kept_words: list[Word]
    speeds: tuple[float, ...] = ()      # one per range; empty is all 1

    def speed(self, i: int) -> float:
        return self.speeds[i] if self.speeds else 1.0

    @property
    def length(self) -> float:
        return sum((b - a) / self.speed(i) for i, (a, b) in enumerate(self.ranges))

    def film_spans(self) -> list[tuple[float, float]]:
        """Where each kept stretch plays in the film."""
        out, at = [], 0.0
        for i, (a, b) in enumerate(self.ranges):
            n = (b - a) / self.speed(i)
            out.append((round(at, 3), round(at + n, 3)))
            at += n
        return out

    def film_speech(self) -> list[tuple[float, float]]:
        """When someone speaks in the film: each kept word moved to where its stretch plays."""
        out = []
        for i, ((a, b), (at, _)) in enumerate(zip(self.ranges, self.film_spans())):
            v = self.speed(i)
            out += [(at + (w.start - a) / v, at + (w.end - a) / v) for w in self.kept_words
                    if a <= w.start and w.end <= b]
        return out


def with_shows(cut: TalkCut, shows: list[tuple[float, float]], longest: float = 4.0,
               speeds: tuple[float, float] = (2.0, 4.0)) -> TalkCut:
    """The cut with silent stretches of someone showing something put back, sped up so each lasts at most `longest`
    seconds of film (between `speeds`; a caller trims a stretch too long for that). Each must lie between ranges."""
    pieces = [(a, b, cut.speed(i)) for i, (a, b) in enumerate(cut.ranges)]
    for a, b in shows:
        if any(a < y and x < b for x, y, _ in pieces):
            raise ValueError(f"{a}–{b} overlaps what the cut keeps")
        pieces.append((a, b, round(min(max((b - a) / longest, speeds[0]), speeds[1]), 2)))
    pieces.sort()
    return TalkCut([(a, b) for a, b, _ in pieces], cut.removed, cut.kept_words, tuple(v for *_, v in pieces))


def cut_talk(words: list[Word], duration: float, loud: np.ndarray | None = None, rate: int = 100,
             **kw) -> TalkCut:
    """The jump cut of a recording. One where nobody speaks keeps nothing: all of it is a silent stretch, for
    `silent_shows` to put back what is worth seeing."""
    if not within(words, 0, duration):
        return TalkCut([], [Removed(0.0, round(duration, 3), "pause")], [])
    if loud is not None:
        words = tighten(words, loud, rate)
    ranges = keep_ranges(words, 0, duration, **kw)
    if loud is not None:
        ranges = snap_to_quiet(ranges, words, loud, rate)
    kept = [w for w in within(words, 0, duration) if any(a <= w.start and w.end <= b for a, b in ranges)]
    flags = {k: kw[k] for k in ("drop_fillers", "drop_repeats") if k in kw}
    return TalkCut(ranges, removed(words, 0, duration, ranges, **flags), kept)


def clip_id(stem: str, i: int) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_-]+', '-', stem).strip('-') or 'talk'}-{i}"


@dataclass(frozen=True)
class Part:
    """One file's cut, as the film plays it."""
    src: str            # relative to film.html
    stem: str
    cut: TalkCut
    speech_sound: str | None = None     # levelled copies of the file's sound, played instead of its own:
    ambient_sound: str | None = None    # one for the stretches with speech, one for those without

    def speaks(self, i: int) -> bool:
        a, b = self.cut.ranges[i]
        return any(a <= w.start and w.end <= b for w in self.cut.kept_words)

    def sound(self, i: int) -> str | None:
        return self.speech_sound if self.speaks(i) else self.ambient_sound


SPEECH_LUFS = -16.0     # what short-video platforms play speech at
AMBIENT_LUFS = -23.0    # a clip without speech sits under it
CEILING_DBTP = -1.0


def measure(path: Path, ranges: list[tuple[float, float]]) -> tuple[float, float]:
    """Integrated loudness (LUFS, EBU R128) and true peak (dBTP) of the stretches of a file the cut keeps."""
    # atrim + concat, not aselect: older ffmpeg builds (Ubuntu's 6.x) select whole audio frames by aselect's `t`
    # and measured every stretch alike.
    if ranges:
        n = len(ranges)
        cuts = "".join(f"[s{k}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[c{k}];"
                       for k, (a, b) in enumerate(ranges))
        graph = (f"[0:a]asplit={n}" + "".join(f"[s{k}]" for k in range(n)) + ";" + cuts
                 + "".join(f"[c{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1,ebur128=peak=true[out]")
    else:
        graph = "[0:a]ebur128=peak=true[out]"
    err = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-filter_complex", graph, "-map", "[out]",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    pk = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    num = lambda m: float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")  # noqa: E731
    return num(i), num(pk)


def gain(lufs: float, speech: bool, most: float = 30.0) -> float:
    """The gain in dB that brings a clip to the film's level: speech to SPEECH_LUFS (by at most `most`), a clip
    without speech to the quieter AMBIENT_LUFS and raised by at most 10 dB (its hiss is not worth hearing). A clip
    with no measurable sound is left as it is. Peaks are not its concern: `level` puts a limiter after it, since a
    phone's quiet speech often comes with a pan's clatter at full scale, which a gain alone could not raise past."""
    if not np.isfinite(lufs) or lufs < -70:
        return 0.0
    db = (SPEECH_LUFS if speech else AMBIENT_LUFS) - lufs
    return round(min(db, most if speech else 10.0), 2)


def level_to(src: Path, dst: Path, ranges: list[tuple[float, float]], speech: bool, length: float,
             tries: int = 3) -> tuple[float, float, float]:
    """`level` the file so the kept stretches reach the film's level: a loud instant (a pan's clatter) counts in
    the first measure and makes it read louder than the speech is, but the limiter takes it out, so the gain is
    measured again after limiting and corrected. Returns the gain in dB, and the loudness and true peak after."""
    lufs, _ = measure(src, ranges)
    db = gain(lufs, speech)
    target, most = (SPEECH_LUFS, 30.0) if speech else (AMBIENT_LUFS, 10.0)
    after, peak = lufs, float("-inf")
    for _ in range(tries):
        level(src, dst, db, length)
        after, peak = measure(dst, ranges)
        if not np.isfinite(after) or abs(target - after) < 0.5 or db >= most:
            break
        db = round(min(db + target - after, most), 2)
    return db, after, peak


def level(src: Path, dst: Path, db: float, length: float) -> None:
    """The file's sound raised by `db` with a limiter at CEILING_DBTP, as a WAV padded with silence to `length`,
    the video's (a phone's sound track can end a few ms before its picture), so the same #t= plays the same
    moment and never runs past the file."""
    limit = 10 ** ((CEILING_DBTP - 1.0) / 20)      # the limiter is not true-peak: leave room for it
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-af",
                    f"volume={db}dB,alimiter=limit={limit:.3f}:attack=5:release=50:level=false,apad=whole_dur={length}",
                    "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(dst)], check=True)


def reel_speech(parts: list[Part]) -> list[tuple[float, float]]:
    """When someone speaks across the whole film."""
    out, at = [], 0.0
    for p in parts:
        out += [(at + a, at + b) for a, b in p.cut.film_speech()]
        at += p.cut.length
    return out


def film_html(parts: list[Part], size: tuple[int, int], music_src: str | None = None,
              music_volume: float = 0.5) -> str:
    """film.html: every part's kept stretches one after another on a track, and the music, if any, below."""
    tags, sounds, at = [], [], 0.0
    for p in parts:
        for i, ((a, b), (t, _)) in enumerate(zip(p.cut.ranges, p.cut.film_spans())):
            v = p.cut.speed(i)
            when = f'at="{round(at + t, 3):g}"' + (f' speed="{v:g}"' if v != 1 else "")
            sound = p.sound(i)
            tags.append(f'  <video id="{clip_id(p.stem, i + 1)}" src="{escape(p.src)}#t={a:g},{b:g}" {when}'
                        + (' volume="0"' if sound else "") + "></video>")
            if sound:
                sounds.append(f'  <audio id="{clip_id(p.stem, i + 1)}-sound" src="{escape(sound)}#t={a:g},{b:g}" '
                              f"{when}></audio>")
        at += p.cut.length
    clips = "\n".join(tags) + ("\n</section>\n<section>\n" + "\n".join(sounds) if sounds else "")
    music = (f'\n<section>\n  <audio id="music" src="{escape(music_src)}#t=0,{at:.3f}" '
             f'volume="{music_volume:g}"></audio>\n</section>') if music_src else ""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width={size[0]}, height={size[1]}">
</head>
<body>
<section>
{clips}
</section>{music}
</body>
</html>
"""


def ducked_music(music: np.ndarray, length: float, speech: list[tuple[float, float]], level: float = 0.25,
                 fade_out: float = 1.5) -> np.ndarray:
    """The music `length` seconds long, looped if it is shorter, lowered under `speech` and faded out at the end."""
    n = int(round(length * RATE))
    reps = int(np.ceil(n / max(len(music), 1)))
    out = np.tile(music, (reps, 1))[:n].astype(np.float32)
    t = np.arange(n) / RATE
    gain = duck_gain(t, speech, level=level)
    gain *= np.clip((length - t) / fade_out, 0, 1)
    return out * gain[:, None].astype(np.float32)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def report(cut: TalkCut, duration: float, shows: list = (), fixes: list = ()) -> str:
    counts = {why: [r for r in cut.removed if r.why == why] for why in ("pause", "filler", "repeat", "retake", "adjust")}
    lines = [f"{duration:.1f}s → {cut.length:.1f}s  (−{duration - cut.length:.1f}s, {len(cut.ranges)} clips)",
             "  cut: " + ", ".join(f"{len(v)} {k}{'s' if len(v) != 1 else ''} ({sum(r.end - r.start for r in v):.1f}s)"
                                   for k, v in counts.items())]
    for r in cut.removed:
        if r.why != "pause":
            lines.append(f"  {r.start:7.2f}–{r.end:<7.2f} {r.why:<7} {r.text}")
    if shows:
        lines.append("  silent stretches: " + ", ".join(
            f"{s.kind} {s.start:.1f}–{s.end:.1f}" + (f" ({s.what})" if getattr(s, "what", "") else "") for s in shows))
        for i, (a, b) in enumerate(cut.ranges):
            if not any(a <= w.start and w.end <= b for w in cut.kept_words):
                v = cut.speed(i)
                lines.append(f"  {a:7.2f}–{b:<7.2f} shown" + (f" at {v:g}x ({(b - a) / v:.1f}s)" if v != 1 else ""))
    for f in fixes:
        lines.append(f"  subtitle  {f.wrong} → {f.right}   {f.why}")
    return "\n".join(lines)


def load_env(path: Path) -> None:
    """KEY=VALUE lines from a .env file into the environment, without overriding what is set."""
    for line in path.read_text().splitlines():
        k, eq, v = line.strip().partition("=")
        if eq and k and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def subtract(ranges: list[tuple[float, float]], holes: list[tuple[float, float]], words: list[Word],
             shortest: float = 0.3) -> list[tuple[float, float]]:
    """`ranges` with `holes` taken out, except where someone speaks: a hole that has words in it is cut only
    before its first word and after its last, never between words (a cut a few ms long between two words is a
    click, not a cut). Pieces left shorter than `shortest` are dropped."""
    cuts = []
    for a, b in holes:
        said = [w for w in words if w.end > a and w.start < b]
        if not said:
            cuts.append((a, b))
            continue
        first, last = min(w.start for w in said), max(w.end for w in said)
        cuts += [(x, y) for x, y in ((a, first), (last, b)) if y - x >= shortest]
    out = list(ranges)
    for x, y in cuts:
        nxt = []
        for a, b in out:
            if y <= a or x >= b:
                nxt.append((a, b))
                continue
            if x > a:
                nxt.append((a, x))
            if y < b:
                nxt.append((y, b))
        out = nxt
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a >= shortest]


def without_adjusting(cut: TalkCut, segments: list, ends_only: float | None = None) -> TalkCut:
    """The cut without the moments the camera was being picked up or moved (`segments` of kind "adjust"). With
    `ends_only` (the clip's length), only those at its start or end: in a short shot the camera moving in the
    middle is the shot (a walk, a pan), while at its ends it is the phone being raised or lowered."""
    holes = [(s.start, s.end) for s in segments if s.kind == "adjust"]
    if ends_only is not None:
        holes = [(a, b) for a, b in holes if a <= 0.3 or b >= ends_only - 0.3]
    if not holes or cut.speeds:
        return cut
    ranges = subtract(cut.ranges, holes, cut.kept_words)
    gone = [Removed(a, b, "adjust") for a, b in holes if any(a < y and x < b for x, y in cut.ranges)]
    return TalkCut(ranges, sorted(cut.removed + gone, key=lambda r: r.start), cut.kept_words)


def judged_shows(gaps: list[tuple[float, float]], segments: list, longest: float = 3.0,
                 shortest: float = 1.0) -> list[tuple[float, float]]:
    """From the silent stretches, what is worth seeing: the best few seconds of each "show" segment in them, at
    most `longest`, never less than `shortest`, at its own speed. Filler and camera moves are left out."""
    out = []
    for g0, g1 in gaps:
        for s in segments:
            if s.kind != "show":
                continue
            a, b = s.best or (s.start, s.end)
            a, b = max(a, g0, s.start), min(b, g1, s.end)
            if b - a > longest:
                mid = (a + b) / 2
                a, b = mid - longest / 2, mid + longest / 2
            if b - a >= shortest:
                out.append((round(a, 3), round(b, 3)))
    return sorted(out)


def silences(ranges: list[tuple[float, float]], words: list[Word], duration: float, shortest: float = 2.0,
             pad: float = 0.15) -> list[tuple[float, float]]:
    """The stretches the cut leaves out in which nobody says anything at all (not even a filler or a first try),
    at least `shortest` long: where something worth seeing may be put back."""
    out = subtract([(0.0, duration)], ranges, [], 0.0)
    for w in words:
        out = subtract(out, [(w.start - pad, w.end + pad)], [], 0.0)
    return [(a, b) for a, b in out if b - a >= shortest]


def silent_shows(video: Path, cut: TalkCut, mode: str, longest: float, shortest_gap: float = 2.0,
                 segments: list | None = None, words: list[Word] | None = None, duration: float | None = None):
    """The silent stretches worth putting back ("show": someone showing something; "keep": all of them, as they
    are), and what each silent stretch was judged to be."""
    if words is not None and duration is not None:
        gaps = silences(cut.ranges, words, duration, shortest_gap)
    else:
        gaps = [(r.start, r.end) for r in cut.removed if r.why == "pause" and r.end - r.start >= shortest_gap]
    if mode == "cut" or not gaps:
        return cut, []
    if mode == "keep":
        return with_shows(cut, gaps, speeds=(1.0, 1.0)), []
    if segments is not None:            # judged by a model that watched: shows at their own speed
        holes = [(s.start, s.end) for s in segments if s.kind == "adjust"]
        shows = [r for a, b in judged_shows(gaps, segments, longest) for r in subtract([(a, b)], holes, [], 0.8)]
        return with_shows(cut, shows, speeds=(1.0, 1.0)), [s for s in segments if any(
            s.start < b and a < s.end for a, b in gaps)]
    from .frames import FrameSampler
    from .silent import SilentClassifier, YuNetFaces

    clip = FrameSampler(rate=5).sample(video)
    judge = SilentClassifier(YuNetFaces())
    judged = [s for a, b in gaps for s in judge.classify(clip, a, b)]
    shows = []
    for s in judged:
        if s.kind != "show" or s.length < 1.0:
            continue
        a, b = (s.start, s.end) if s.length <= longest * 4 else judge.busiest(clip, s.start, s.end, longest * 4)
        shows.append((a, b))
    return with_shows(cut, shows, longest), judged


VIDEO = {".mov", ".mp4", ".m4v"}


def studio_has(project: Path, port: int = 4747) -> bool:
    """Whether OpenFilm Studio is running and has had this project open. Studio merges a write of film.html with
    what it takes for the person's edits, matched by clip id; a new cut reuses ids for other stretches, so the
    merge scrambles it (clips moved onto others, a film 100 s too long)."""
    import socket

    if not (project / ".film" / "history").exists():
        return False
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def videos_in(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO) if p.is_dir() else [p]
    return [p.resolve() for p in out]


def timeline(groups: list[list[Shot]], short: float) -> str:
    lines = []
    for k, group in enumerate(groups, 1):
        lines.append(f"scene {k}")
        for s in group:
            when = local(s.probe.shot_at).strftime("%m-%d %H:%M:%S") if s.probe.shot_at else "  (no time)  "
            kind = "b-roll, kept whole" if s.probe.duration < short else "talk"
            lines.append(f"  {when}  {s.path.name:<18}{s.probe.duration:7.1f}s  {kind}")
    return "\n".join(lines)


def main(transcriber: Transcriber | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, nargs="+", help="videos, or folders of them")
    ap.add_argument("--out", type=Path, help="the OpenFilm project folder (default: ~/Movies/OpenFilm/Projects/<name>-talk)")
    ap.add_argument("--music", type=Path, help="music to lay under the speech, lowered while someone talks")
    ap.add_argument("--music-volume", type=float, default=0.5)
    ap.add_argument("--duck", type=float, default=0.25, help="music gain under speech (default 0.25, about −12 dB)")
    ap.add_argument("--language", help="e.g. zh, en (default: detected)")
    ap.add_argument("--asr", choices=["paraformer", "whisper"],
                    help="speech recognition: paraformer (Mandarin, the `asr` extra; default with --language zh) "
                         "or whisper (the `speech` extra)")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="the Whisper model")
    ap.add_argument("--max-pause", type=float, default=0.4, help="pauses longer than this are cut (seconds)")
    ap.add_argument("--keep-fillers", action="store_true")
    ap.add_argument("--keep-repeats", action="store_true")
    ap.add_argument("--silent", choices=["show", "cut", "keep"], default="show",
                    help="stretches nobody speaks in: show = keep the best seconds of what is worth seeing (default); "
                         "cut = drop them all; keep = keep them all as they are")
    ap.add_argument("--watch", choices=["gemini", "local"], default="gemini",
                    help="who judges the picture: gemini watches each recording (uploads a 360 px copy; also finds "
                         "camera moves), local guesses from faces and motion and speeds shows up (nothing uploaded)")
    ap.add_argument("--show-longest", type=float, default=3.0, help="seconds a shown stretch lasts at most")
    ap.add_argument("--story", default="", help="what the film is about, in your words: tells the model watching "
                    "what is worth keeping (e.g. 穿着海鸥衣服去公司，晚上去鬼屋)")
    ap.add_argument("--days", nargs="+", metavar="MM-DD", help="the order of the days in the film (default: "
                    "the order they happened), e.g. --days 11-02 10-30 11-01")
    ap.add_argument("--short", type=float, default=8.0, help="clips shorter than this are b-roll, kept whole")
    ap.add_argument("--scene-gap", type=float, default=10.0, help="minutes between shots that start a new scene")
    ap.add_argument("--fix", action="store_true", help="fix misheard subtitle words with Gemini (the `llm` extra)")
    ap.add_argument("--env-file", type=Path, help="a .env with GEMINI_API_KEY")
    ap.add_argument("--notes", action="store_true", help="add a notes track to write requests on in Studio "
                    "(notes already on the film are always kept, moved to where their moment now plays)")
    ap.add_argument("--force", action="store_true", help="write over the project's film.html")
    args = ap.parse_args()

    shots = in_day_order([Shot(p, probe(p)) for p in videos_in(args.videos)], args.days or [])
    slips = [s for s in shots if s.probe.duration < 0.5]       # a tap on the record button, not a shot
    shots = [s for s in shots if s.probe.duration >= 0.5]
    if not shots:
        raise SystemExit("no videos")
    if slips:
        print("left out (under half a second): " + ", ".join(s.path.name for s in slips))
    name = args.videos[0].resolve().name if len(args.videos) == 1 else shots[0].path.stem
    out = args.out or Path.home() / "Movies" / "OpenFilm" / "Projects" / f"{Path(name).stem}-talk"
    film = out / "film.html"
    if film.exists() and not args.force:
        raise SystemExit(f"{film} exists: pass --force to write over it")
    if film.exists() and studio_has(out):
        raise SystemExit(f"OpenFilm Studio is running and has opened {out.name}: it would merge this cut with its "
                         "own copy and scramble it. Quit Studio first, or write a new project with --out.")
    if args.env_file:
        load_env(args.env_file)
    sizes = {(s.probe.width, s.probe.height) for s in shots}
    size = max(sizes, key=lambda z: sum((s.probe.width, s.probe.height) == z for s in shots))
    groups = scenes(shots, args.scene_gap * 60)
    print(timeline(groups, args.short))
    if len(sizes) > 1:
        print(f"  shots differ in size {sorted(sizes)}: the film is {size[0]}x{size[1]}, the others fit inside it")

    if transcriber is None:
        if (args.asr or ("paraformer" if args.language == "zh" else "whisper")) == "paraformer":
            from .speech import ParaformerTranscriber
            transcriber = ParaformerTranscriber()
        else:
            from .speech import WhisperTranscriber
            transcriber = WhisperTranscriber(args.model, args.language)
    judge = None
    if args.watch == "gemini":
        from .gemini import GeminiSceneJudge
        try:
            judge = GeminiSceneJudge(story=args.story)
            judge.medium(shots[0].path)        # fails early without a key
        except RuntimeError as e:
            print(f"  judging the picture locally: {e}")
            judge = None
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / ".film").mkdir(exist_ok=True)
    parts, log = [], []
    for shot in shots:
        video, info = shot.path, shot.probe
        asset = out / "assets" / re.sub(r"\s+", "_", video.name)
        link_or_copy(video, asset)
        stem = asset.stem
        segments = None
        if judge is not None:
            print(f"\n{video.name}: watching ({info.duration:.1f}s)…")
            try:
                segments = judge.segments(video, info.duration)
            except RuntimeError as e:
                print(f"  could not watch it: {e}")
        if info.duration < args.short:
            cut = TalkCut([(0.0, info.duration)], [], [])
            if segments:
                cut = without_adjusting(cut, segments, ends_only=info.duration)
                cut = TalkCut([r for r in cut.ranges if r[1] - r[0] >= 1.0] or cut.ranges, cut.removed, [])
                if cut.removed:
                    print(f"{video.name}: b-roll, camera moves cut: "
                          + ", ".join(f"{r.start:.1f}–{r.end:.1f}" for r in cut.removed))
            parts.append(Part(f"assets/{asset.name}", stem, cut))
            continue
        print(f"{video.name}: transcribing ({info.duration:.1f}s)…")
        words = CachedTranscriber(transcriber, CACHE).transcribe(video)
        cut = cut_talk(words, info.duration, loudness(read_audio(video, RATE), RATE), max_pause=args.max_pause,
                       drop_fillers=not args.keep_fillers, drop_repeats=not args.keep_repeats)
        if segments:
            cut = without_adjusting(cut, segments)
        cut, judged = silent_shows(video, cut, args.silent, args.show_longest, segments=segments, words=words,
                                   duration=info.duration)
        lines, fixes = subtitle_lines(cut.kept_words), []
        if args.fix and lines:
            from .correct import CachedCorrector, GeminiCorrector, apply_fixes
            try:
                if judge is not None:
                    from .gemini import GeminiListeningCorrector
                    found, alike = GeminiListeningCorrector(judge.medium(video)).fixes([l.text for l in lines]), 0.35
                else:
                    found, alike = CachedCorrector(GeminiCorrector(), CACHE).fixes([l.text for l in lines]), 0.6
            except RuntimeError as e:
                print(f"  subtitles left as heard: {e}")
                found, alike = [], 0.6
            lines, fixes = apply_fixes(lines, found, alike)
        note = "Made by aicut-talk: fillers and repeated words left out" + (", misheard words fixed." if args.fix else ".")
        asset.with_suffix(".vtt").write_text(to_vtt(lines, note))
        parts.append(Part(f"assets/{asset.name}", stem, cut))
        log.append({"file": video.name, "ranges": cut.ranges, "speeds": cut.speeds,
                    "removed": [r.__dict__ for r in cut.removed], "silent": [s.__dict__ for s in judged],
                    "fixes": [f.__dict__ for f in fixes]})
        print(report(cut, info.duration, judged, fixes))

    print("\nlevelling the sound…")
    durations = {f"assets/{re.sub(r'\s+', '_', s.path.name)}": s.probe.duration for s in shots}
    leveled = []
    for p in parts:
        if not p.cut.ranges:
            leveled.append(p)
            continue
        found = {}
        for speech in (True, False):
            ranges = [r for i, r in enumerate(p.cut.ranges) if p.speaks(i) == speech]
            if not ranges:
                continue
            name = f"assets/{p.stem}.{'speech' if speech else 'ambient'}.wav"
            lufs, peak = measure(out / p.src, ranges)
            db, after, top = level_to(out / p.src, out / name, ranges, speech, durations[p.src])
            found[speech] = name
            print(f"  {p.stem:<12}{'speech' if speech else 'ambient':<8}{lufs:7.1f} LUFS  peak {peak:6.1f}  "
                  f"{db:+6.1f} dB  →  {after:6.1f} LUFS  peak {top:5.1f}")
        leveled.append(Part(p.src, p.stem, p.cut, found.get(True), found.get(False)))
    parts = leveled
    length = sum(p.cut.length for p in parts)
    music_src = None
    if args.music:
        music_src = f"assets/{args.music.stem}-ducked.wav"
        write_wav(out / music_src, ducked_music(read_audio(args.music, RATE), length, reel_speech(parts), args.duck),
                  RATE)
    text = film_html(parts, size, music_src, args.music_volume)
    old = film.read_text() if film.exists() else ""
    from .notes import Note, clips_of, keep_apart, place, read_notes, track, write_page
    notes = keep_apart([Note(place(n.at, clips_of(old), clips_of(text)), n.length, n.text) for n in read_notes(old)])
    if args.notes and not notes:
        notes = [Note(0.0, 3.0, "@双击写要求")]
    if notes:
        write_page(out)
        text = text.replace("<body>\n", "<body>\n" + track(notes, size), 1)
        print(f"\n{len(notes)} note{'s' if len(notes) != 1 else ''} on the film")
    film.write_text(text)
    (out / ".film" / "talk-cut.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str))
    total = sum(s.probe.duration for s in shots)
    print(f"\n{len(shots)} shots, {total:.1f}s → {length:.1f}s")
    print(f"→ {film}")


if __name__ == "__main__":
    main()
