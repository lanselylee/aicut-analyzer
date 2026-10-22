"""Sets and reps: where a person repeats a movement, how many times, and where each repetition turns.

Nothing here needs to see the person. A camera on a tripod watching someone repeat a movement sees the picture
oscillate, so the picture itself is the signal: frames shrunk to 24×24, reduced to their first principal component
(the direction in which the picture changes most). That works from any angle, with half the body out of frame,
where a pose model loses the person.

Two steps:
- `SetFinder` slides a window along the clip and asks whether the picture repeats itself there (autocorrelation of
  the first component); long stretches that do are sets.
- `RepCounter` looks inside one set: the turns of the signal are the moments the movement stops and reverses (the
  bar at the top of a curl, back at rest), and a rep goes from one turn of a kind to the next.

Turns matter beyond counting: a cut lands well where the movement is still, not halfway through it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from .frames import Clip

TINY = 24


def tiny_frames(clip: Clip) -> np.ndarray:
    """Each frame as TINY×TINY gray values in [0, 1], one row per frame."""
    return np.array([cv2.resize(f.gray, (TINY, TINY), interpolation=cv2.INTER_AREA).ravel() for f in clip.frames],
                    np.float64) / 255


def principal_axis(x: np.ndarray) -> np.ndarray:
    """The direction in which the rows of `x` vary most."""
    return np.linalg.svd(x - x.mean(0), full_matrices=False)[2][0]


def first_component(x: np.ndarray) -> np.ndarray:
    """Rows of `x` projected on their first principal component."""
    p = x @ principal_axis(x)
    return p - p.mean()


def smooth(s: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k)
    return np.convolve(np.pad(s, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")


def periodicity(s: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, int]:
    """How strongly `s` repeats itself (normalized autocorrelation, ≤ 1) and at which lag.

    A smooth signal correlates with itself at small lags just because it changes slowly, so only a peak that comes
    after the autocorrelation has dipped below zero counts: the signal went away and came back.
    """
    s = s - s.mean()
    if s.var() < 1e-12:
        return 0.0, 0
    ac = np.correlate(s, s, "full")[s.size - 1:] / (s.var() * s.size)
    dipped = np.flatnonzero(ac < 0)
    if dipped.size == 0:
        return 0.0, 0
    lags = [k for k in range(max(min_lag, int(dipped[0]) + 1), min(max_lag, s.size - 2))
            if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not lags:
        return 0.0, 0
    k = max(lags, key=lambda k: ac[k])
    return float(ac[k]), k


@dataclass(frozen=True)
class Rep:
    start: float    # a turn: the movement leaves this position…
    turn: float     # …stops and reverses at the other end…
    end: float      # …and is back where it started
    likeness: float = 1.0   # how closely it traces the set's usual rep (correlation of the two curves, ≤ 1)


@dataclass
class WorkSet:
    start: float
    end: float
    period: float           # seconds per rep
    reps: list[Rep]
    times: np.ndarray       # the set's sampled frames…
    position: np.ndarray    # …where the movement is, 0 at one end of its range and 1 at the other…
    speed: np.ndarray       # …and how fast it moves, as a share of its usual top speed (can exceed 1)

    def turns(self) -> list[float]:
        """Every moment the movement stops and reverses, in order."""
        return sorted({r.start for r in self.reps} | {r.turn for r in self.reps} | {r.end for r in self.reps})

    def still_at(self, t: float) -> float:
        """How fast the movement is at time `t` (share of top speed): low means a cut can land there."""
        return float(np.interp(t, self.times, self.speed))


class SetFinder:
    """Stretches of a clip where the picture keeps repeating itself.

    A frame belongs to a stretch when any window around it repeats. That overshoots the set by up to a window, on
    purpose: `RepCounter` trims each stretch to its first and last turn, and a set missed here is lost for good.
    """

    def __init__(self, window_s: float = 8.0, hop_s: float = 1.0, min_period_s: float = 0.8,
                 max_period_s: float = 5.0, threshold: float = 0.45, min_set_s: float = 4.0):
        self.window_s = window_s        # long enough for two reps of the slowest movement
        self.hop_s = hop_s
        self.min_period_s = min_period_s
        self.max_period_s = max_period_s
        self.threshold = threshold      # autocorrelation that counts as repeating
        self.min_set_s = min_set_s

    def find(self, x: np.ndarray, fps: float) -> list[tuple[int, int, float]]:
        """(first frame, frame after the last, period in seconds) of each set in tiny frames `x`."""
        n = len(x)
        win, hop = int(self.window_s * fps), max(1, int(self.hop_s * fps))
        lo, hi = int(self.min_period_s * fps), int(self.max_period_s * fps)
        active = np.zeros(n + 1, bool)
        periods: list[tuple[int, int]] = []
        for a in range(0, max(1, n - win + 1), hop):
            r, lag = periodicity(smooth(first_component(x[a:a + win]), 3), lo, hi)
            if r > self.threshold:
                active[a:a + win] = True
                periods.append((a, lag))
        sets = []
        start = None
        for i, on in enumerate(active):
            if on and start is None:
                start = i
            elif not on and start is not None:
                if i - start >= self.min_set_s * fps:
                    lags = [lag for a, lag in periods if start <= a < i]
                    sets.append((start, i, float(np.median(lags)) / fps))
                start = None
        return sets


class RepCounter:
    """The reps inside one stretch of repetition, each split at its turns.

    Not every turn of the signal is a rep's: fidgeting before a set has turns too. A set's turns alternate between
    the two ends of the movement at a steady pace and a steady size, so only runs of such turns count.
    """

    def __init__(self, prominence: float = 0.35, margin_s: float = 5.0, context_s: float = 10.0):
        self.prominence = prominence    # share of the movement's range a turn must stand out by
        self.margin_s = margin_s        # turns are looked for this far beyond the stretch's edges: the set finder's
                                        # window (8 s) blurs them by up to half its length
        self.context_s = context_s      # and the rest position this far, where the person is surely still

    def count(self, x: np.ndarray, fps: float, a: int, b: int, period: float) -> list[WorkSet]:
        """The sets in frames a…b of tiny frames `x`: one, or several where the reps stop or turn irregular."""
        m, c = int(self.margin_s * fps), int(self.context_s * fps)
        core = x[a:b]
        ca, cb = max(0, a - c), min(len(x), b + c)
        a, b = max(0, a - m), min(len(x), b + m)
        # the axis is found where the picture repeats, not in the margin, where bigger movements (picking up the bar,
        # sitting down) would take it over; and on frames with their slow drift removed (light changing, the person
        # shifting on the seat). Projecting the frames as they are keeps the position meaningful
        drift = np.column_stack([smooth(core[:, j], int(2 * period * fps) | 1) for j in range(core.shape[1])])
        axis = principal_axis(core - drift)
        offset = (core @ axis).mean()
        s = smooth(x[a:b] @ axis - offset, 3)
        context = smooth(x[ca:cb] @ axis - offset, 3)
        inside = core @ axis
        spread = np.percentile(inside, 95) - np.percentile(inside, 5)
        if spread <= 0:
            return []
        gap = max(1, int(0.3 * period * fps))
        tops, _ = find_peaks(s, distance=gap, prominence=self.prominence * spread)
        bottoms, _ = find_peaks(-s, distance=gap, prominence=self.prominence * spread)
        if not tops.size or not bottoms.size:
            return []
        turns = self._alternating(s, self._centered(s, tops, spread), self._centered(-s, bottoms, spread))
        t = (a + np.arange(len(s))) / fps
        speed = np.abs(np.gradient(s))
        lo, hi = np.percentile(s, 5), np.percentile(s, 95)
        sets = []
        for run in self._regular_runs(s, turns, period * fps):
            rest_kind = self._rest_kind(context, a - ca, s, run, fps)
            rest = np.array([i for i, k in run if k == rest_kind], int)
            far = np.array([i for i, k in run if k != rest_kind], int)
            if not far.size:
                continue
            rest = self._with_edges(s, rest, far, spread)
            reps = []
            for r0, r1 in zip(rest, rest[1:]):
                mid = far[(far > r0) & (far < r1)]
                if mid.size == 1:
                    reps.append((int(r0), int(mid[0]), int(r1)))
            reps = self._trimmed(s, reps)
            if not reps:
                continue
            top_speed = np.percentile(speed[reps[0][0]:reps[-1][2] + 1], 95)
            like = self._likeness(s, reps) if len(reps) >= 3 else [1.0] * len(reps)
            sets.append(WorkSet(start=float(t[reps[0][0]]), end=float(t[reps[-1][2]]),
                                period=float(np.median([(r1 - r0) / fps for r0, _, r1 in reps])),
                                reps=[Rep(float(t[r0]), float(t[m]), float(t[r1]), round(lk, 3))
                                      for (r0, m, r1), lk in zip(reps, like)],
                                times=t, position=np.clip((s - lo) / (hi - lo), 0, 1),
                                speed=speed / max(top_speed, 1e-12)))
        return sets

    @staticmethod
    def _centered(s: np.ndarray, peaks: np.ndarray, spread: float) -> np.ndarray:
        """Each peak moved to the middle of the plateau around it (within 3 % of the range of its value).

        A movement often pauses at its turns (a squeeze at the top of a curl), and the picture barely changes there;
        the turn is the middle of that pause, not wherever noise puts the highest sample.
        """
        out = []
        for p in peaks:
            flat = s >= s[p] - 0.03 * spread
            i = j = p
            while i > 0 and flat[i - 1]:
                i -= 1
            while j < s.size - 1 and flat[j + 1]:
                j += 1
            out.append((i + j) // 2)
        return np.array(out, int)

    @staticmethod
    def _alternating(s: np.ndarray, tops: np.ndarray, bottoms: np.ndarray) -> list[tuple[int, int]]:
        """Tops (+1) and bottoms (-1) in time order; of two of a kind in a row, the more extreme one."""
        out: list[tuple[int, int]] = []
        for i, k in sorted([(int(i), 1) for i in tops] + [(int(i), -1) for i in bottoms]):
            if out and out[-1][1] == k:
                if k * s[i] > k * s[out[-1][0]]:
                    out[-1] = (i, k)
            else:
                out.append((i, k))
        return out

    @staticmethod
    def _regular_runs(s: np.ndarray, turns: list[tuple[int, int]], period: float) -> list[list[tuple[int, int]]]:
        """Runs of turns a steady half rep apart (15–85 % of a period, as lifting and lowering can differ) and of
        a steady size (at least half the usual swing between turns): at least one whole rep's worth."""
        if len(turns) < 2:
            return []
        swing = [abs(s[j] - s[i]) for (i, _), (j, _) in zip(turns, turns[1:])]
        usual = np.median(swing)
        runs, run = [], [turns[0]]
        for (i, _), (j, k), sw in zip(turns, turns[1:], swing):
            if 0.15 * period <= j - i <= 0.85 * period and sw >= 0.5 * usual:
                run.append((j, k))
            else:
                runs.append(run)
                run = [(j, k)]
        runs.append(run)
        return [r for r in runs if len(r) >= 2]

    @staticmethod
    def _curve(s: np.ndarray, r: tuple[int, int, int]) -> np.ndarray:
        """A rep's stretch of the signal resampled to 20 points and standardized, to compare reps of any length."""
        c = np.interp(np.linspace(r[0], r[2], 20), np.arange(s.size), s)
        return (c - c.mean()) / (c.std() + 1e-12)

    @classmethod
    def _likeness(cls, s: np.ndarray, reps: list[tuple[int, int, int]]) -> list[float]:
        """Each rep's correlation with the set's usual rep (the median curve)."""
        usual = np.median([cls._curve(s, r) for r in reps], axis=0)
        usual = (usual - usual.mean()) / (usual.std() + 1e-12)
        return [float(np.mean(cls._curve(s, r) * usual)) for r in reps]

    @classmethod
    def _trimmed(cls, s: np.ndarray, reps: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """The reps without the odd ones at either end. Putting the bar back right after the last rep, or settling in
        before the first, moves like a rep but not like this set's reps: over 40 % longer or shorter than the usual
        rep, under 60 % of its swing, or tracing another curve (likeness under 0.6). An odd rep in the middle stays:
        it is a slow rep.

        On real footage a set's reps reach a likeness of 0.9 or more, its first and last ones 0.5 to 0.8 (a shorter
        range, letting go of the handle), random movement 0.4: an edge rep near the line may go, which costs little,
        as an edit takes its reps from the middle of a set."""
        def odd(r, like, length, swing):
            return abs((r[2] - r[0]) - length) > 0.4 * length or abs(s[r[1]] - s[r[0]]) < 0.6 * swing or like < 0.6

        while len(reps) >= 3:
            like = cls._likeness(s, reps)
            length = np.median([r[2] - r[0] for r in reps])
            swing = np.median([abs(s[r[1]] - s[r[0]]) for r in reps])
            if odd(reps[-1], like[-1], length, swing):
                reps = reps[:-1]
            elif odd(reps[0], like[0], length, swing):
                reps = reps[1:]
            else:
                break
        return reps

    @staticmethod
    def _rest_kind(context: np.ndarray, at: int, s: np.ndarray, run: list[tuple[int, int]], fps: float) -> int:
        """Which end of the movement is rest (+1 tops, -1 bottoms): a rep starts and ends there.

        Rest is the position the person holds still next to the set: the nearest half second or more of (nearly) no
        movement within two seconds before the run's first turn or after its last, in `context` (the same signal
        over a wider stretch; `s` starts at its frame `at`). Fidgeting is movement, so it does not count. With no
        still stretch on either side, rest is the kind of the first turn.
        """
        first, last = at + run[0][0], at + run[-1][0]
        speed = np.abs(np.gradient(context))
        still = np.append(speed < 0.1 * np.percentile(speed[first:last + 1], 95), False)
        reach = 2 * fps
        near, i = [], 0
        while i < still.size:
            j = i
            while still[j]:
                j += 1
            if j - i >= 0.5 * fps and (first - reach <= j <= first + 1 or last - 1 <= i <= last + reach):
                near.append((i, j))
            i = j + 1
        if not near:
            return run[0][1]
        level = np.median(np.concatenate([context[i:j] for i, j in near]))
        tops = np.mean([s[i] for i, k in run if k == 1])
        bottoms = np.mean([s[i] for i, k in run if k == -1])
        return 1 if abs(level - tops) < abs(level - bottoms) else -1

    @staticmethod
    def _with_edges(s: np.ndarray, rest: np.ndarray, far: np.ndarray, spread: float) -> np.ndarray:
        """Rest turns plus the moments the first rep leaves rest and the last one settles back.

        After a still stretch the signal is flat, so leaving rest is not a peak and the first rep would be lost;
        likewise the last. Those moments are where the signal, within 15 % of the movement's range of rest, stops
        getting any closer to it.
        """
        level = np.median(s[rest]) if rest.size else s[0]
        dist = np.abs(s - level)
        near = dist < 0.15 * spread
        settled = 0.01 * spread
        out = list(rest)
        if not (rest < far[0]).any():
            before = np.flatnonzero(near[:far[0]])
            if before.size:
                i = int(before[-1])
                while i > 0 and dist[i - 1] < dist[i] - settled:
                    i -= 1
                out.insert(0, i)
        if not (rest > far[-1]).any():
            after = np.flatnonzero(near[far[-1]:])
            if after.size:
                i = int(far[-1] + after[0])
                while i < s.size - 1 and dist[i + 1] < dist[i] - settled:
                    i += 1
                out.append(i)
        return np.array(out, int)


class RepAnalyzer:
    """Sets and reps of a sampled clip (Facade over the two steps)."""

    def __init__(self, finder: SetFinder | None = None, counter: RepCounter | None = None, min_reps: int = 3):
        self.finder = finder or SetFinder()
        self.counter = counter or RepCounter()
        self.min_reps = min_reps    # two similar movements in a row happen by chance (reaching, adjusting)

    def analyze(self, clip: Clip) -> list[WorkSet]:
        x = tiny_frames(clip)
        sets = []
        for a, b, period in self.finder.find(x, clip.fps):
            for ws in self.counter.count(x, clip.fps, a, b, period):
                if len(ws.reps) >= self.min_reps and not (sets and ws.start < sets[-1].end):
                    sets.append(ws)
        return sets
