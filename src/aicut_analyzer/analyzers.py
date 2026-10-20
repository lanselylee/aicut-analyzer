"""Quality analyzers. Each one turns a sampled clip into a per-frame score in [0, 1], 1 being best.

They share one interface (Strategy): the scorer combines whichever analyzers it is given, so adding a new signal
(faces, smiles, duplicates…) means adding a class, not changing the scorer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .frames import Clip


class Analyzer(ABC):
    name: str

    @abstractmethod
    def raw(self, clip: Clip) -> np.ndarray:
        """The measured value per frame, in the analyzer's own unit (for reports)."""

    @abstractmethod
    def score(self, raw: np.ndarray) -> np.ndarray:
        """Raw values mapped to [0, 1], 1 being best."""


class SharpnessAnalyzer(Analyzer):
    """Variance of the Laplacian: low when the picture is out of focus or smeared by motion blur.

    Its absolute value mostly measures how much detail the scene has (bare branches against the sky score 6000, a
    selfie with a blurred background 76, both in focus), so each frame is judged against the sharpest frames around
    it: a blurred frame falls far below its neighbours. The reference is local (`reach` frames either side), not the
    whole clip's, because a walking shot changes scene: pavement has far less detail than the trees it reaches.
    """
    name = "sharpness"

    def __init__(self, reference_pct: float = 90, sharp_ratio: float = 0.6, reach: int = 15):
        self.reference_pct = reference_pct   # the neighbourhood's "sharp" level
        self.sharp_ratio = sharp_ratio       # at this share of that level or above, a frame counts as fully sharp
        self.reach = reach                   # frames either side (1 s at 15 fps)

    def raw(self, clip: Clip) -> np.ndarray:
        return np.array([cv2.Laplacian(f.gray, cv2.CV_64F).var() for f in clip.frames])

    def score(self, raw: np.ndarray) -> np.ndarray:
        ref = np.array([np.percentile(raw[max(0, i - self.reach):i + self.reach + 1], self.reference_pct)
                        for i in range(raw.size)])
        return np.clip(raw / (self.sharp_ratio * np.maximum(ref, 1e-6)), 0, 1)


class StabilityAnalyzer(Analyzer):
    """Hand shake: the jitter of the camera's motion around its smoothed path.

    The global shift between frames comes from phase correlation. A steady pan moves a lot but smoothly, so it is
    not shake; what is left after smoothing the path over `smooth_s` seconds is.
    """
    name = "stability"

    def __init__(self, smooth_s: float = 0.5, shaky_at: float = 6.0):
        self.smooth_s = smooth_s
        self.shaky_at = shaky_at   # px of jitter per frame (at 360 px) that counts as fully shaky

    def raw(self, clip: Clip) -> np.ndarray:
        frames = clip.frames
        if len(frames) < 2:
            return np.zeros(len(frames))
        window = cv2.createHanningWindow(frames[0].gray.shape[::-1], cv2.CV_32F)
        shifts = [(0.0, 0.0)]
        for a, b in zip(frames, frames[1:]):
            (dx, dy), _ = cv2.phaseCorrelate(a.gray.astype(np.float32), b.gray.astype(np.float32), window)
            shifts.append((dx, dy))
        path = np.cumsum(np.array(shifts), axis=0)
        k = max(3, int(self.smooth_s * clip.fps) | 1)
        kernel = np.ones(k) / k
        smooth = np.column_stack([np.convolve(np.pad(path[:, i], k // 2, mode="edge"), kernel, "valid")
                                  for i in range(2)])
        jitter = np.linalg.norm(path - smooth, axis=1)
        return np.abs(np.diff(jitter, prepend=jitter[0])) + 0.5 * jitter

    def score(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(1 - raw / self.shaky_at, 0, 1)


class ExposureAnalyzer(Analyzer):
    """Clipped highlights and crushed shadows: the share of pixels near pure white or pure black."""
    name = "exposure"

    def __init__(self, bad_at: float = 0.25):
        self.bad_at = bad_at   # this share of clipped pixels or more counts as badly exposed

    def raw(self, clip: Clip) -> np.ndarray:
        out = []
        for f in clip.frames:
            v = cv2.cvtColor(f.bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
            out.append(float(np.mean((v >= 250) | (v <= 8))))
        return np.array(out)

    def score(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(1 - raw / self.bad_at, 0, 1)
