"""What happens while nobody speaks: someone showing something to the camera, a close-up of hands at work, walking
off to fetch something, or standing still. A talking-head cut keeps the first (sped up) and drops the others.

- `FaceSizer` says how wide the nearest face is, as a share of the frame (`YuNetFaces` runs OpenCV's YuNet face
  detector; its 230 KB model is downloaded once to ~/.cache/aicut/models).
- `SilentClassifier` labels each sampled frame of a silent stretch: near the camera (a face at least `near` of the
  frame wide) and moving → "show", near and still → "still"; no near face but the picture busy (a close-up of a pan,
  hands at work) → "show", no near face and the picture calm (an empty room, someone far off) → "away". It then
  splits the stretch where the label changes for long enough.

Measured on a kitchen vlog: presenting to the camera, faces are 0.22–0.39 of the frame wide; walking to the fridge,
none is found or one under 0.06. Without a face, a frying-egg close-up moves 10 (median, 0–255) between frames five
times a second, an empty kitchen or someone at the far fridge 0.2–1.0.
"""
from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frames import Clip, Frame

MODELS = Path.home() / ".cache" / "aicut" / "models"
YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class FaceSizer(ABC):
    @abstractmethod
    def width(self, frame: Frame) -> float:
        """The nearest face's width as a share of the frame's width; 0 without a face."""


class YuNetFaces(FaceSizer):
    def __init__(self, model: Path | None = None, score: float = 0.7):
        if model is None:
            model = MODELS / Path(YUNET).name
            if not model.exists():
                model.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(YUNET, model)
        self.det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score)

    def width(self, frame: Frame) -> float:
        h, w = frame.bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame.bgr)
        return 0.0 if faces is None else float(max(f[2] for f in faces)) / w


@dataclass(frozen=True)
class Stretch:
    start: float
    end: float
    kind: str           # "show", "still" or "away"
    motion: float       # mean change between sampled frames, 0–255

    @property
    def length(self) -> float:
        return self.end - self.start


def motion(a: Frame, b: Frame) -> float:
    """How much the picture changed between two frames: mean absolute difference of tiny grayscale copies."""
    small = lambda f: cv2.resize(f.gray, (48, 64 if f.gray.shape[0] > f.gray.shape[1] else 36),  # noqa: E731
                                 interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(small(a) - small(b)).mean())


def runs(labels: list[str], times: list[float], shortest: float) -> list[tuple[int, int, str]]:
    """Runs of the same label as (first, last + 1, label), a run shorter than `shortest` seconds folded into the run
    before it (or after, at the start)."""
    out: list[list] = []
    for i, lab in enumerate(labels):
        if out and out[-1][2] == lab:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1, lab])
    merged: list[list] = []
    for r in out:
        span = times[min(r[1], len(times) - 1)] - times[r[0]]
        if merged and (span < shortest or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and times[merged[0][1]] - times[0] < shortest:
        merged[1][0] = 0
        merged.pop(0)
    return [tuple(r) for r in merged]


class SilentClassifier:
    def __init__(self, faces: FaceSizer, near: float = 0.15, still: float = 2.0, busy: float = 5.0,
                 shortest: float = 1.5):
        self.faces = faces
        self.near = near
        self.still = still
        self.busy = busy
        self.shortest = shortest

    def classify(self, clip: Clip, start: float, end: float) -> list[Stretch]:
        frames = [f for f in clip.frames if start <= f.t < end]
        if len(frames) < 2:
            return [Stretch(start, end, "still", 0.0)]
        moves = [motion(a, b) for a, b in zip(frames, frames[1:])]
        moves = [moves[0], *moves]
        near = [self.faces.width(f) >= self.near for f in frames]
        # motion is judged over about a second, so one twitch does not make a still stretch move
        k = max(1, round(clip.fps))
        smooth = np.convolve(moves, np.ones(k) / k, mode="same")
        labels = [("show" if m >= self.still else "still") if n else ("show" if m >= self.busy else "away")
                  for n, m in zip(near, smooth)]
        times = [f.t for f in frames]
        out = []
        for a, b, lab in runs(labels, times, self.shortest):
            s = start if a == 0 else times[a]
            e = end if b >= len(frames) else times[b]
            out.append(Stretch(round(s, 3), round(e, 3), lab, round(float(np.mean(moves[a:b])), 2)))
        return out

    def busiest(self, clip: Clip, start: float, end: float, length: float) -> tuple[float, float]:
        """The `length` seconds of [start, end] where the picture moves most."""
        frames = [f for f in clip.frames if start <= f.t < end]
        if end - start <= length or len(frames) < 3:
            return start, min(end, start + length)
        moves = np.array([0.0] + [motion(a, b) for a, b in zip(frames, frames[1:])])
        times = np.array([f.t for f in frames])
        best, at = -1.0, start
        for i, t in enumerate(times):
            if t + length > end:
                break
            m = moves[(times >= t) & (times < t + length)].sum()
            if m > best:
                best, at = m, t
        return round(float(at), 3), round(float(at) + length, 3)
