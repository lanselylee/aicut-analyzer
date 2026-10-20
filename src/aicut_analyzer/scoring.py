"""Combining analyzers into one quality curve per clip, and picking the best stretch of it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyzers import Analyzer
from .frames import Clip


@dataclass
class ClipReport:
    name: str
    duration: float
    times: np.ndarray
    signals: dict[str, np.ndarray]       # analyzer name → per-frame score
    raw: dict[str, np.ndarray]           # analyzer name → per-frame raw value
    quality: np.ndarray                  # weighted per-frame score
    best: tuple[float, float] | None = None
    best_quality: float = 0.0
    usable: bool = False
    notes: list[str] = field(default_factory=list)

    def mean_over(self, a: float, b: float, key: str | None = None) -> float:
        curve = self.quality if key is None else self.signals[key]
        m = (self.times >= a) & (self.times < b)
        return float(curve[m].mean()) if m.any() else float("nan")


class QualityScorer:
    """A weighted mean of the analyzers' scores, frame by frame."""

    def __init__(self, weighted: list[tuple[Analyzer, float]]):
        self.weighted = weighted

    def report(self, clip: Clip) -> ClipReport:
        signals, raw = {}, {}
        for analyzer, _ in self.weighted:
            r = analyzer.raw(clip)
            raw[analyzer.name] = r
            signals[analyzer.name] = analyzer.score(r)
        total = sum(w for _, w in self.weighted)
        quality = sum(signals[a.name] * w for a, w in self.weighted) / total
        return ClipReport(name=clip.path.name, duration=clip.duration, times=np.array([f.t for f in clip.frames]),
                          signals=signals, raw=raw, quality=quality)


class WindowPicker:
    """The best stretch of a clip: high mean quality, no bad frame inside, longer is better up to `max_len`."""

    def __init__(self, min_len: float = 1.0, max_len: float = 5.0, floor: float = 0.35, usable_at: float = 0.55,
                 length_bonus: float = 0.03):
        self.min_len, self.max_len = min_len, max_len
        self.floor = floor              # no frame in the window may score below this
        self.usable_at = usable_at      # a best window under this mean is not worth using
        self.length_bonus = length_bonus

    def pick(self, rep: ClipReport) -> ClipReport:
        q, t = rep.quality, rep.times
        n = len(q)
        if n < 2:
            rep.notes.append("too short to judge")
            return rep
        dt = t[1] - t[0]
        lo = max(2, round(self.min_len / dt))
        hi = min(n, round(self.max_len / dt))
        best, best_val = None, -1.0
        csum = np.concatenate([[0.0], np.cumsum(q)])
        for length in range(lo, hi + 1):
            for i in range(0, n - length + 1):
                seg_min = q[i:i + length].min()
                if seg_min < self.floor:
                    continue
                mean = (csum[i + length] - csum[i]) / length
                val = mean + self.length_bonus * length * dt
                if val > best_val:
                    best, best_val = (i, length), val
        if best is None:
            rep.notes.append(f"no {self.min_len:g}s stretch without a bad frame")
            return rep
        i, length = best
        a, b = float(t[i]), float(min(rep.duration, t[i] + length * dt))
        rep.best = (round(a, 2), round(b, 2))
        rep.best_quality = float(q[i:i + length].mean())
        rep.usable = rep.best_quality >= self.usable_at
        for key, sig in rep.signals.items():
            if rep.mean_over(0, rep.duration, key) < 0.5:
                rep.notes.append(f"weak {key} overall")
        return rep
