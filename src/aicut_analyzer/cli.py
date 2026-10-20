"""aicut-analyze <folder> [--film film.html] [--json out.json] [--plot out.png] [--clips NAME ...]

Scores every clip in a folder and proposes its best stretch. With --film, compares the proposals with the stretches an
OpenFilm film.html already uses (its `#t=in,out`), to see where the numbers and the eye disagree. --plot draws the
quality curves (needs the `plot` extra); --clips limits the run to some clips.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .analyzers import ExposureAnalyzer, SharpnessAnalyzer, StabilityAnalyzer
from .frames import FrameSampler
from .scoring import ClipReport, QualityScorer, WindowPicker

VIDEO = {".mov", ".mp4", ".m4v"}


def film_picks(film: Path) -> dict[str, tuple[float, float]]:
    """{file name: (in, out)} for every video clip in an OpenFilm film.html."""
    picks = {}
    for src in re.findall(r'<video[^>]*\bsrc="([^"]+)"', film.read_text()):
        m = re.match(r"(.+?)#t=([\d.]+),([\d.]+)", src)
        if m:
            picks.setdefault(Path(m.group(1)).name, (float(m.group(2)), float(m.group(3))))
    return picks


def to_json(rep: ClipReport) -> dict:
    return {"clip": rep.name, "duration": round(rep.duration, 2), "best": rep.best,
            "best_quality": round(rep.best_quality, 3), "usable": rep.usable, "notes": rep.notes,
            "overall": {k: round(rep.mean_over(0, rep.duration, k), 3) for k in rep.signals},
            "curve": {"t": [round(x, 2) for x in rep.times],
                      **{k: [round(float(x), 3) for x in v] for k, v in rep.signals.items()}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--film", type=Path, help="an OpenFilm film.html to compare against")
    ap.add_argument("--json", type=Path, help="write the full report, curves included")
    ap.add_argument("--plot", type=Path, help="draw the quality curves to this PNG")
    ap.add_argument("--clips", nargs="+", metavar="NAME", help="only these clips (file names)")
    args = ap.parse_args()

    sampler = FrameSampler()
    scorer = QualityScorer([(SharpnessAnalyzer(), 0.45), (StabilityAnalyzer(), 0.40), (ExposureAnalyzer(), 0.15)])
    picker = WindowPicker()
    picks = film_picks(args.film) if args.film else {}

    reports = []
    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in VIDEO)
    if args.clips:
        paths = [p for p in paths if p.name in args.clips]
    for path in paths:
        rep = picker.pick(scorer.report(sampler.sample(path)))
        reports.append(rep)

    head = f"{'clip':<14}{'dur':>6}  {'sharp':>5} {'stable':>6} {'expo':>5}   {'algo pick':<13}{'q':>5}"
    if picks:
        head += f"   {'film pick':<13}{'q':>5}"
    print(head)
    print("-" * len(head))
    for rep in reports:
        o = {k: rep.mean_over(0, rep.duration, k) for k in rep.signals}
        best = f"{rep.best[0]:.1f}–{rep.best[1]:.1f}" if rep.best else "—"
        line = (f"{rep.name:<14}{rep.duration:>5.1f}s  {o['sharpness']:>5.2f} {o['stability']:>6.2f} {o['exposure']:>5.2f}"
                f"   {best:<13}{rep.best_quality:>5.2f}{'' if rep.usable else ' ✗'}")
        if picks:
            p = picks.get(rep.name)
            line += f"   {f'{p[0]:.1f}–{p[1]:.1f}' if p else '—':<13}{rep.mean_over(*p) if p else float('nan'):>5.2f}"
        if rep.notes:
            line += "   " + "; ".join(rep.notes)
        print(line)

    if args.json:
        args.json.write_text(json.dumps([to_json(r) for r in reports], ensure_ascii=False, indent=1))
        print(f"\n→ {args.json}")
    if args.plot:
        from .plot import plot_reports
        print(f"→ {plot_reports(reports, args.plot, picks)}")


if __name__ == "__main__":
    main()
