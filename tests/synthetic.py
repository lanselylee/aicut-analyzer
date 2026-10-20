"""Synthetic footage with known defects: a textured scene seen through a moving crop, with blur, shake or clipping
added where a test wants them."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aicut_analyzer.frames import Clip, Frame

W, H = 360, 640          # portrait, like phone footage at the sampler's size
FPS = 15.0


def scene(seed: int = 0, size: tuple[int, int] = (1400, 1400)) -> np.ndarray:
    """A detailed BGR picture: noise at several scales plus edges, so sharpness and phase correlation have texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((*size, 3), np.float32)
    for scale in (4, 16, 64):
        small = rng.random((size[0] // scale, size[1] // scale, 3)).astype(np.float32)
        img += cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)
    img = img / 3
    for _ in range(60):
        x, y = rng.integers(0, size[1]), rng.integers(0, size[0])
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(20, 120)), int(y + rng.integers(20, 120))),
                      tuple(float(c) for c in rng.random(3)), 2)
    return np.clip(img * 200 + 25, 0, 255).astype(np.uint8)   # mid-range: no clipping by itself


def path(n: int, pan: tuple[float, float] = (0.0, 0.0), shake: float = 0.0, seed: int = 1) -> np.ndarray:
    """Camera top-left corner per frame: a steady pan of `pan` px/frame plus random jitter of `shake` px."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n)[:, None] * np.array(pan)[None, :]
    return 300 + steps + rng.normal(0, shake, (n, 2)) if shake else 300 + steps


def footage(n: int = 60, pan=(0.0, 0.0), shake: float = 0.0, blur: slice | None = None,
            clip_white: slice | None = None, seed: int = 0) -> list[np.ndarray]:
    """`n` BGR frames: the scene through a moving window, blurred and/or overexposed over the given frame ranges."""
    img = scene(seed)
    frames = []
    for i, (x, y) in enumerate(path(n, pan, shake)):
        f = img[int(round(y)):int(round(y)) + H, int(round(x)):int(round(x)) + W].copy()
        if blur and blur.start <= i < blur.stop:
            f = cv2.GaussianBlur(f, (0, 0), 6)
        if clip_white and clip_white.start <= i < clip_white.stop:
            f = cv2.convertScaleAbs(f, alpha=3.0, beta=120)
        frames.append(f)
    return frames


def as_clip(frames: list[np.ndarray], name: str = "synthetic.mp4") -> Clip:
    """A sampled clip straight from frames, without encoding a file."""
    return Clip(path=Path(name), duration=len(frames) / FPS, fps=FPS,
                frames=[Frame(t=i / FPS, gray=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), bgr=f) for i, f in enumerate(frames)])


def write(frames: list[np.ndarray], path: Path, fps: float = 30.0) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        out.write(f)
    out.release()
    return path
