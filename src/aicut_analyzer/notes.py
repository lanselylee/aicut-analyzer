"""Notes the person writes on the film for Claude: a track of sticky notes over the picture, each placed over the
stretch it is about and holding a request ("@这里删掉", "@加个音效"). They are read back from film.html, and they
survive a new cut: each is pinned to the moment of the recording it was placed over, and put back wherever that
moment plays in the new film.

In Studio, a note is the page notes/note.html on the track marked `data-aicut-notes`; double-clicking it edits its
text (kept as an `overrides` entry on #note), and splitting or trimming it places it. Notes are not part of the
film: the track is hidden from a render.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

MARK = 'data-aicut-notes="1"'
PAGE = "notes/note.html"
_TRACK = re.compile(r'\n?<section[^>]*\bdata-aicut-notes="1"[^>]*>.*?</section>', re.S)
_ATTR = re.compile(r'([\w-]+)=("([^"]*)"|\'([^\']*)\')')

NOTE_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: transparent; }
  #note {
    position: absolute; inset: 0; box-sizing: border-box; padding: 14px 20px;
    border-radius: 16px; background: rgba(255, 214, 102, 0.92); color: #222;
    border: 4px dashed #d9480f;
    font: 600 34px/1.3 "PingFang SC", -apple-system, sans-serif;
    display: flex; align-items: center; justify-content: center; text-align: center;
  }
</style>
</head>
<body>
<div id="note">@双击写要求</div>
<script type="module">
// an editing note for Claude, not part of the final film
window.film = { width: 560, height: 160, frame() {} };
</script>
</body>
</html>
"""


@dataclass(frozen=True)
class Note:
    at: float           # where it starts on the film
    length: float
    text: str


@dataclass(frozen=True)
class Clip:
    """A video clip as film.html places it."""
    src: str
    source_in: float
    source_out: float
    at: float
    speed: float = 1.0

    @property
    def end(self) -> float:
        return self.at + (self.source_out - self.source_in) / self.speed

    def to_source(self, t: float) -> float:
        return self.source_in + (t - self.at) * self.speed

    def to_film(self, s: float) -> float:
        return self.at + (s - self.source_in) / self.speed


def attrs(tag: str) -> dict[str, str]:
    return {m.group(1): html.unescape(m.group(3) if m.group(3) is not None else m.group(4))
            for m in _ATTR.finditer(tag)}


def fragment(src: str) -> tuple[str, float | None, float | None]:
    m = re.match(r"(.+?)#t=([\d.]+)(?:,([\d.]+))?$", src)
    if not m:
        return src, None, None
    return m.group(1), float(m.group(2)), float(m.group(3)) if m.group(3) else None


def clips_of(film: str) -> list[Clip]:
    """The video clips of a film.html, in film order."""
    out = []
    for tag in re.findall(r"<video\b[^>]*>", film):
        a = attrs(tag)
        src, i, o = fragment(a.get("src", ""))
        if i is None or o is None:
            continue
        out.append(Clip(src, i, o, float(a.get("at", 0)), float(a.get("speed", 1))))
    return sorted(out, key=lambda c: c.at)


def read_notes(film: str) -> list[Note]:
    """The notes on the notes track, in film order, with the text the person wrote."""
    m = _TRACK.search(film)
    if not m:
        return []
    out = []
    for tag in re.findall(r"<iframe\b[^>]*>", m.group(0)):
        a = attrs(tag)
        _, i, o = fragment(a.get("src", ""))
        text = "@双击写要求"
        for entry in json.loads(a.get("overrides", "[]") or "[]"):
            if entry.get("at") == "#note" and "text" in entry:
                text = entry["text"]
        out.append(Note(float(a.get("at", 0)), round((o or 0) - (i or 0), 3), text))
    return sorted(out, key=lambda n: n.at)


def pin(t: float, clips: list[Clip]) -> tuple[str, float] | None:
    """The recording and its second that play at film time `t`."""
    for c in clips:
        if c.at - 1e-6 <= t < c.end - 1e-6:
            return c.src, c.to_source(t)
    return None


def place(t: float, old: list[Clip], new: list[Clip]) -> float:
    """Where film time `t` of the old cut is in the new one: the same moment of the same recording if the new cut
    keeps it, else the nearest moment of that recording it keeps, else `t` itself."""
    where = pin(t, old)
    if where is None:
        return t
    src, s = where
    same = [c for c in new if c.src == src]
    for c in same:
        if c.source_in - 1e-6 <= s <= c.source_out + 1e-6:
            return round(c.to_film(s), 3)
    if not same:
        return t
    near = min(same, key=lambda c: min(abs(s - c.source_in), abs(s - c.source_out)))
    return round(near.to_film(min(max(s, near.source_in), near.source_out)), 3)


def track(notes: list[Note], stage: tuple[int, int]) -> str:
    """The notes track, to go first in film.html (on top of the picture)."""
    left, top = (stage[0] - 560) // 2, 40
    tags = []
    for i, n in enumerate(sorted(notes, key=lambda n: n.at), 1):
        ov = html.escape(json.dumps([{"at": "#note", "text": n.text}], ensure_ascii=False), quote=False)
        ov = ov.replace("'", "&#39;")
        tags.append(f'  <iframe id="note-{i}" src="{PAGE}#t=0,{n.length:g}" at="{n.at:g}" '
                    f"style=\"left: {left}px; top: {top}px\" overrides='{ov}'></iframe>")
    return f"<section {MARK}>\n" + "\n".join(tags) + "\n</section>\n"


def keep_apart(notes: list[Note]) -> list[Note]:
    """Notes moved so none overlaps the one before (clips on a track may not overlap)."""
    out: list[Note] = []
    for n in sorted(notes, key=lambda n: n.at):
        at = max(n.at, out[-1].at + out[-1].length) if out else n.at
        out.append(Note(round(at, 3), n.length, n.text))
    return out


def write_page(project: Path) -> None:
    page = project / PAGE
    if not page.exists():
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(NOTE_PAGE)


def main() -> None:
    """aicut-notes <project>: the notes on a film, each with the recording and second it was placed over."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("project", type=Path)
    args = ap.parse_args()
    film = (args.project / "film.html").read_text()
    clips = clips_of(film)
    for n in read_notes(film):
        a, b = pin(n.at, clips), pin(n.at + n.length - 0.01, clips)
        where = f"{Path(a[0]).name} {a[1]:.1f}s" if a else "—"
        if b and a and b[0] == a[0]:
            where += f"–{b[1]:.1f}s"
        elif b:
            where += f" … {Path(b[0]).name} {b[1]:.1f}s"
        print(f"{n.at:7.2f}–{n.at + n.length:<7.2f} {where:<32} {n.text}")


if __name__ == "__main__":
    main()
