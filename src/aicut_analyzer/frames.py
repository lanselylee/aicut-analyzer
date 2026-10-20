"""Sampling frames from a clip: small grayscale + color frames at a fixed rate, enough to judge quality cheaply."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    t: float            # seconds from the clip's start
    gray: np.ndarray    # downscaled, for sharpness and motion
    bgr: np.ndarray     # downscaled, for exposure and color


@dataclass(frozen=True)
class Clip:
    path: Path
    duration: float
    fps: float          # sampling rate actually used
    frames: list[Frame]


class FrameSampler:
    """Decodes a clip and keeps `rate` frames per second, scaled so the short side is `short_side` px."""

    def __init__(self, rate: float = 15.0, short_side: int = 360):
        self.rate = rate
        self.short_side = short_side

    def sample(self, path: Path) -> Clip:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / self.rate))
        frames: list[Frame] = []
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                small = self._shrink(img)
                frames.append(Frame(t=i / src_fps, gray=cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), bgr=small))
            i += 1
        cap.release()
        return Clip(path=path, duration=i / src_fps, fps=src_fps / step, frames=frames)

    def _shrink(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        k = self.short_side / min(h, w)
        return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
