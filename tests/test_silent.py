import numpy as np

from aicut_analyzer.frames import Clip, Frame
from aicut_analyzer.silent import FaceSizer, SilentClassifier, runs

FPS = 5


class Faces(FaceSizer):
    """Face width read off the frame: its top-left pixel holds the width × 100."""

    def width(self, frame):
        return frame.bgr[0, 0, 0] / 100


def clip(*parts):
    """("show" | "still" | "away" | "closeup", seconds) parts: show moves near the camera, still stays near, away has
    no face and does not move, closeup has no face and moves."""
    rng = np.random.default_rng(0)
    frames, t = [], 0.0
    base = rng.integers(0, 255, (64, 36)).astype(np.uint8)
    for kind, secs in parts:
        for _ in range(int(secs * FPS)):
            gray = rng.integers(0, 255, (64, 36)).astype(np.uint8) if kind in ("show", "closeup") else base
            bgr = np.zeros((64, 36, 3), np.uint8)
            bgr[0, 0, 0] = 0 if kind in ("away", "closeup") else 30
            frames.append(Frame(round(t, 3), gray, bgr))
            t += 1 / FPS
    return Clip(None, t, FPS, frames)


def kinds(c, start=0.0, end=None):
    return [(s.kind, round(s.start), round(s.end)) for s in SilentClassifier(Faces()).classify(c, start, end or c.duration)]


class TestClassify:
    def test_showing_something_near_the_camera_is_a_show(self):
        assert kinds(clip(("show", 6))) == [("show", 0, 6)]

    def test_walking_off_is_away_and_coming_back_to_show_is_a_show(self):
        assert kinds(clip(("away", 8), ("show", 4))) == [("away", 0, 8), ("show", 8, 12)]

    def test_a_busy_close_up_without_a_face_is_a_show(self):
        # a pan seen from above: no face, but the picture keeps changing
        assert kinds(clip(("closeup", 6))) == [("show", 0, 6)]

    def test_standing_still_near_the_camera_is_still(self):
        assert kinds(clip(("still", 6))) == [("still", 0, 6)]

    def test_a_short_flicker_does_not_split_a_stretch(self):
        assert [k for k, *_ in kinds(clip(("away", 5), ("still", 0.6), ("away", 5)))] == ["away"]

    def test_only_the_asked_stretch_is_judged(self):
        assert kinds(clip(("away", 5), ("show", 5)), 6, 10) == [("show", 6, 10)]


class TestBusiest:
    def test_the_window_where_the_picture_moves_most(self):
        c = clip(("still", 5), ("show", 3), ("still", 5))
        a, b = SilentClassifier(Faces()).busiest(c, 0, 13, 3)
        assert 4.5 <= a <= 5.5 and b - a == 3


class TestRuns:
    def test_runs_of_the_same_label(self):
        assert runs(list("aabbb"), [0, 1, 2, 3, 4], 0.5) == [(0, 2, "a"), (2, 5, "b")]

    def test_a_short_run_folds_into_the_one_before(self):
        assert runs(list("aaaabaaaa"), list(range(9)), 2.0) == [(0, 9, "a")]
