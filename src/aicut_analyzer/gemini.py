"""Gemini watching and listening to a recording (the `llm` extra, its key in GEMINI_API_KEY).

A small copy of the recording (360 px, mono sound) is uploaded once and asked about twice:
- `GeminiSceneJudge` splits it into what happens: talking to the camera, showing something worth seeing (with its
  best few seconds), filler (looking around, searching, walking off), and the camera being picked up or moved. Its
  times are good to about half a second; a caller cuts at speech and quiet moments it finds itself.
- `GeminiListeningCorrector` hears the speech and fixes the subtitles where they differ from what was said, rather
  than guessing from the text alone.
Both are cached by the file and the question, so a recording is asked about once.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from .correct import Corrector, Fix, loads_first, parse_fixes

MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest")
CACHE = Path.home() / ".cache" / "aicut" / "gemini"


def file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


class GeminiMedia:
    """One recording, uploaded once as a small copy, asked any number of questions."""

    def __init__(self, path: Path, api_key: str | None = None, fps: float = 2.0):
        self.path = path
        self.key = file_key(path)
        self.fps = fps
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (or pass --env-file)")
        self._client = None
        self._file = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _upload(self):
        if self._file is None:
            with tempfile.TemporaryDirectory() as tmp:
                small = Path(tmp) / "small.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.path), "-vf",
                                "scale='if(gt(iw,ih),-2,360)':'if(gt(iw,ih),360,-2)'", "-r", "10", "-c:v", "libx264",
                                "-crf", "32", "-preset", "veryfast", "-c:a", "aac", "-ac", "1", "-b:a", "48k",
                                str(small)], check=True)
                f = self.client.files.upload(file=str(small), config={"mime_type": "video/mp4"})
            while f.state and f.state.name == "PROCESSING":
                time.sleep(2)
                f = self.client.files.get(name=f.name)
            if f.state and f.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini could not read {self.path.name}: {f.state.name}")
            self._file = f
        return self._file

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        """The answer, as JSON text (shaped by `schema` when given), from the first model that gives one."""
        from google.genai import errors, types

        f = self._upload()
        video = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                           video_metadata=types.VideoMetadata(fps=self.fps))
        last = None
        for model in MODELS:
            try:
                out = self.client.models.generate_content(
                    model=model, contents=types.Content(role="user", parts=[video, types.Part(text=prompt)]),
                    config={"response_mime_type": "application/json", "temperature": 0,
                            **({"response_schema": schema} if schema else {})})
                return out.text
            except errors.APIError as e:
                last = e
        raise RuntimeError(f"no Gemini model answered: {last}")

    def cached(self, name: str, prompt: str, schema: dict | None = None) -> str:
        cache = CACHE / f"{self.key}-{name}-{hashlib.sha1(prompt.encode()).hexdigest()[:10]}.json"
        if cache.exists():
            return cache.read_text()
        text = self.ask(prompt, schema)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
        return text


# ---------------------------------------------------------------- what happens

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    kind: str                   # "talk", "show", "filler" or "adjust"
    best: tuple[float, float] | None = None     # a show's best few seconds
    what: str = ""


class SceneJudge(ABC):
    @abstractmethod
    def segments(self, path: Path, duration: float) -> list[Segment]:
        """What happens in a recording, start to end."""


SCENE_PROMPT = """这是一段手机拍的中文生活 vlog 原始素材，时长 {duration:.1f} 秒。{story}请从头到尾把它分成连续的片段，每段标一种类型：
- "talk"：人在对镜头说话。
- "show"：没说话，但画面值得留：展示食物或物品、做饭/操作的关键步骤、成品特写、有表情的反应、有意拍的空镜或运镜。
- "filler"：没意义的画面：东张西望、找东西、走开又走回来、发呆等待、离开画面后的空场景、重复的无变化动作。
- "adjust"：在调整镜头：拿起或放下手机、重新构图、镜头晃动或甩动、对焦/画面突然模糊。哪怕只有半秒也要单独标出来。
对 "show" 片段，再给出其中最精彩的一小段（best_start、best_end，1.5–3 秒，动作完整，从动作开始到结束）。
判断有没有意义要结合这个 vlog 的故事：和故事有关、好笑、有反差的画面就是 "show"，哪怕动作本身很平常。
每段素材都是作者特意拍的：除非整段都在调整镜头，至少标出一个 "show" 作为这条素材最有代表性的镜头。
时间用秒，保留一位小数，尽量准确到 0.5 秒以内。片段之间不重叠、不留空。
只返回 JSON 数组，每项：{{"start": 秒, "end": 秒, "kind": "talk|show|filler|adjust", "best_start": 秒或null, "best_end": 秒或null, "what": "十字以内描述"}}。
"""

KINDS = {"talk", "show", "filler", "adjust"}
SCENE_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "kind": {"type": "STRING", "enum": sorted(KINDS)},
    "best_start": {"type": "NUMBER", "nullable": True}, "best_end": {"type": "NUMBER", "nullable": True},
    "what": {"type": "STRING"}}, "required": ["start", "end", "kind"]}}


def parse_segments(text: str, duration: float) -> list[Segment]:
    """Segments from the model's answer: unknown kinds and broken items left out, times kept inside the recording,
    a best stretch kept inside its segment."""
    items = loads_first(text)
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            a, b, kind = max(0.0, float(it["start"])), min(duration, float(it["end"])), str(it["kind"])
        except (KeyError, TypeError, ValueError):
            continue
        if kind not in KINDS or b - a <= 0:
            continue
        best = None
        if it.get("best_start") is not None and it.get("best_end") is not None:
            try:
                x, y = max(a, float(it["best_start"])), min(b, float(it["best_end"]))
                best = (x, y) if y - x > 0.3 else None
            except (TypeError, ValueError):
                best = None
        out.append(Segment(round(a, 2), round(b, 2), kind, best, str(it.get("what", ""))))
    return sorted(out, key=lambda s: s.start)


class GeminiSceneJudge(SceneJudge):
    """`story`, the person's own words for what the film is about ("穿着海鸥衣服去公司，晚上去鬼屋"), tells it what
    is worth seeing: a seagull typing at a desk is filler to a model that does not know, and the joke to one that
    does."""

    def __init__(self, api_key: str | None = None, story: str = ""):
        self.api_key = api_key
        self.story = story
        self.media: dict[Path, GeminiMedia] = {}

    def medium(self, path: Path) -> GeminiMedia:
        if path not in self.media:
            self.media[path] = GeminiMedia(path, self.api_key)
        return self.media[path]

    def segments(self, path: Path, duration: float) -> list[Segment]:
        story = f"这个 vlog 讲的是：{self.story}。" if self.story else ""
        prompt = SCENE_PROMPT.format(duration=duration, story=story)
        return parse_segments(self.medium(path).cached("scenes", prompt, SCENE_SCHEMA), duration)


# ---------------------------------------------------------------- what is said

LISTEN_PROMPT = """视频里的人在说中文。下面是自动识别出的字幕，每行前面是行号，可能有听错的字词。
请认真听视频里的声音，把字幕和实际说的话对照：只改听错的字词（同音字、近音字、人名、品牌、食物名等），按实际读音改。
规则：
- 不润色、不改语气、不增删内容，不改标点，不合并或拆分行。
- 听不清的不改。
- wrong 必须是该行原文中一字不差的连续片段，尽量短。
只返回 JSON 数组，每项：{"line": 行号, "wrong": "原文片段", "right": "实际说的", "why": "简短理由"}。没有错就返回 []。

"""


class GeminiListeningCorrector(Corrector):
    """Fixes subtitle lines by listening to the recording they came from."""

    listens = True

    def __init__(self, media: GeminiMedia):
        self.media = media
        self.model = MODELS[0]

    def fixes(self, lines: list[str]) -> list[Fix]:
        prompt = LISTEN_PROMPT + "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
        return parse_fixes(self.media.cached("listen", prompt))
