"""Fixing what speech recognition misheard: a language model reads the subtitles with their context and names the
words it got wrong ("算明证" → "三明治"); the fixes are checked and applied to the timed words, so subtitles keep their
timing.

- `Corrector` is the interface: lines in, fixes out. `GeminiCorrector` asks Gemini (the `llm` extra, its key in
  GEMINI_API_KEY); another model is another corrector. `CachedCorrector` asks once per text.
- `apply_fixes` only takes a fix whose wrong text is in its line, and changes nothing else: a model that rewrites a
  line instead of fixing a word is not followed. `sounds_alike` keeps only fixes that sound like what was heard
  (pinyin), since a mishearing sounds like the word: asked about "算明证" (suàn míng zhèng), a model guessing from
  context alone said "便当盒"; the speaker said 三明治.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from difflib import SequenceMatcher

from .speech import CJK, Line, Word, join_words


@dataclass(frozen=True)
class Fix:
    line: int           # index into the lines asked about
    wrong: str          # as it is in the line
    right: str
    why: str = ""


class Corrector(ABC):
    @abstractmethod
    def fixes(self, lines: list[str]) -> list[Fix]:
        """The misheard words in these subtitle lines, in order."""


PROMPT = """下面是一段中文 vlog 口播的语音识别字幕，每行前面是行号。识别结果里可能有听错的字词：同音或近音字、
专有名词、品牌名、食物名等。请结合上下文，只改正明显的识别错误。
规则：
- 不润色、不改语气、不删不加内容，不改标点，不合并或拆分行。
- 听不清、不确定的不要改。
- 识别错误和原话读音相近：改正后的读音必须和原文片段相近（拼音相似），读音差得远的不要改。
- wrong 必须是该行原文中一字不差的连续片段，尽量短（只包含错的词）。
只返回 JSON 数组，每项：{"line": 行号, "wrong": "原文片段", "right": "改正后", "why": "简短理由"}。没有错就返回 []。

"""


class GeminiCorrector(Corrector):
    """Asks each model in turn until one answers: a busy model (503) is common at peak hours."""

    def __init__(self, models: tuple[str, ...] = ("gemini-flash-latest", "gemini-flash-lite-latest"),
                 api_key: str | None = None):
        self.models = models
        self.model = models[0]
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (or pass --env-file)")

    def fixes(self, lines: list[str]) -> list[Fix]:
        from google import genai
        from google.genai import errors

        client = genai.Client(api_key=self.api_key)
        text = PROMPT + "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
        last = None
        for model in self.models:
            try:
                out = client.models.generate_content(model=model, contents=text, config={
                    "response_mime_type": "application/json", "temperature": 0})
                return parse_fixes(out.text)
            except errors.APIError as e:      # busy or over quota: the next model may not be
                last = e
        raise RuntimeError(f"no Gemini model answered: {last}")


def loads_first(text: str):
    """The first JSON value in a model's answer, ignoring what trails it (a stray "}" or a sentence) and forgiving
    a comma before a closing bracket; None if there is none."""
    text = text.strip()
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        return None
    for attempt in (text[start:], re.sub(r",\s*([\]}])", r"\1", text[start:])):   # then without trailing commas
        try:
            return json.JSONDecoder().raw_decode(attempt)[0]
        except json.JSONDecodeError:
            continue
    return None


def parse_fixes(text: str) -> list[Fix]:
    items = loads_first(text)
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            out.append(Fix(int(it["line"]), str(it["wrong"]), str(it["right"]), str(it.get("why", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


class CachedCorrector(Corrector):
    def __init__(self, inner: Corrector, folder: Path):
        self.inner = inner
        self.folder = folder

    def fixes(self, lines: list[str]) -> list[Fix]:
        key = hashlib.sha1(f"{type(self.inner).__name__}:{getattr(self.inner, 'model', '')}:{PROMPT}\n"
                           f"{chr(10).join(lines)}".encode()).hexdigest()[:16]
        cache = self.folder / f"fixes-{key}.json"
        if cache.exists():
            return [Fix(**f) for f in json.loads(cache.read_text())]
        fixes = self.inner.fixes(lines)
        self.folder.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([asdict(f) for f in fixes], ensure_ascii=False))
        return fixes


def pinyin(text: str) -> str:
    from pypinyin import lazy_pinyin

    return "".join(lazy_pinyin(text)).lower()


def sounds_alike(wrong: str, right: str, alike: float = 0.6) -> bool:
    """Whether a fix sounds like what was heard: toneless pinyin at least `alike` the same (difflib's ratio).
    Text that is not Chinese is compared as it is written."""
    a, b = (pinyin(wrong), pinyin(right)) if CJK.search(wrong + right) else (wrong.lower(), right.lower())
    return SequenceMatcher(None, a, b).ratio() >= alike


def _spans(words: tuple[Word, ...]) -> tuple[str, list[tuple[int, int]]]:
    """A line's text and where each word is in it, as (start, end) character offsets."""
    text, spans = "", []
    for w in words:
        nxt = join_words([text, w.text]) if text else w.text
        spans.append((len(nxt) - len(w.text), len(nxt)))
        text = nxt
    return text, spans


def apply_fixes(lines: list[Line], fixes: list[Fix], alike: float = 0.6) -> tuple[list[Line], list[Fix]]:
    """The lines with each fix made, and the fixes that were. A fix is made where its wrong text first is in its
    line: the words it touches become one word with the right text, from the first one's start to the last one's
    end. A fix whose wrong text is not in its line, which changes nothing, which rewrites more than it fixes (its
    length off by more than half), or which does not sound like what was heard (pinyin less than `alike` the same),
    is left out. A model that listened may be trusted further (0.35: "蒜头卷" → "三明治" is 0.38, while a crowd's
    "没没信" → "再试一回", 0.30, was a guess)."""
    lines = list(lines)
    done = []
    for f in fixes:
        if not (0 <= f.line < len(lines)) or not f.wrong or f.wrong == f.right:
            continue
        if abs(len(f.right) - len(f.wrong)) > max(2, len(f.wrong) // 2):
            continue
        if alike and not sounds_alike(f.wrong, f.right, alike):
            continue
        words = lines[f.line].words
        text, spans = _spans(words)
        at = text.find(f.wrong)
        if at < 0:
            continue
        end = at + len(f.wrong)
        hit = [i for i, (a, b) in enumerate(spans) if a < end and b > at]
        first, last = hit[0], hit[-1]
        a0, b1 = spans[first][0], spans[last][1]
        merged = Word(text[a0:at] + f.right + text[end:b1], words[first].start, words[last].end,
                      min(words[i].prob for i in hit))
        lines[f.line] = Line(words[:first] + (merged,) + words[last + 1:])
        done.append(f)
    return lines, done
