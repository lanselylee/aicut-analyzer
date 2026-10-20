from pathlib import Path

import numpy as np
import pytest

from aicut_analyzer.analyzers import Analyzer
from aicut_analyzer.scoring import ClipReport, QualityScorer, WindowPicker
from synthetic import as_clip, footage


class Fixed(Analyzer):
    """An analyzer whose scores are given, to test the scorer on its own."""

    def __init__(self, name, values):
        self.name, self.values = name, np.asarray(values, float)

    def raw(self, clip):
        return self.values

    def score(self, raw):
        return raw


def report(quality, fps=15.0):
    q = np.asarray(quality, float)
    t = np.arange(q.size) / fps
    return ClipReport(name="c", duration=q.size / fps, times=t, signals={"q": q}, raw={"q": q}, quality=q)


class TestQualityScorer:
    def test_weighted_mean_of_the_analyzers(self):
        clip = as_clip(footage(4))
        scorer = QualityScorer([(Fixed("a", [1, 1, 0, 0]), 3), (Fixed("b", [1, 0, 1, 0]), 1)])
        np.testing.assert_allclose(scorer.report(clip).quality, [1, 0.75, 0.25, 0])

    def test_keeps_every_signal_for_the_report(self):
        rep = QualityScorer([(Fixed("a", [0.5] * 4), 1)]).report(as_clip(footage(4)))
        assert set(rep.signals) == {"a"} and rep.raw["a"].tolist() == [0.5] * 4


class TestWindowPicker:
    def test_picks_the_good_stretch(self):
        q = [0.2] * 15 + [0.95] * 30 + [0.2] * 15          # 1 s bad, 2 s good, 1 s bad
        rep = WindowPicker().pick(report(q))
        assert rep.best == pytest.approx((1.0, 3.0), abs=0.1)
        assert rep.usable

    def test_no_bad_frame_inside_the_window(self):
        q = [0.9] * 30 + [0.1] + [0.9] * 20                # one bad frame splits the clip
        rep = WindowPicker(floor=0.35).pick(report(q))
        a, b = rep.best
        assert b <= 30 / 15 + 1e-6 or a >= 31 / 15 - 1e-6

    def test_prefers_longer_when_quality_is_equal(self):
        rep = WindowPicker(max_len=5).pick(report([0.9] * 90))
        assert rep.best[1] - rep.best[0] == pytest.approx(5, abs=0.1)

    def test_a_clip_that_is_bad_throughout_is_not_usable(self):
        rep = WindowPicker().pick(report([0.4] * 45))
        assert not rep.usable

    def test_says_why_when_nothing_qualifies(self):
        rep = WindowPicker(floor=0.5).pick(report([0.9, 0.1] * 30))
        assert rep.best is None and rep.notes


def test_end_to_end_on_an_encoded_file(tmp_path: Path):
    from aicut_analyzer.analyzers import ExposureAnalyzer, SharpnessAnalyzer, StabilityAnalyzer
    from aicut_analyzer.frames import FrameSampler
    from synthetic import write

    # 4 s at 30 fps: a shaky second, two steady ones, a blurred one
    shaky = footage(30, pan=(1, 1), shake=8)
    steady = footage(60, pan=(1, 1))
    blurred = footage(30, pan=(1, 1), blur=slice(0, 30))
    path = write(shaky + steady + blurred, tmp_path / "clip.mp4")

    clip = FrameSampler().sample(path)
    assert clip.duration == pytest.approx(4.0, abs=0.05)
    scorer = QualityScorer([(SharpnessAnalyzer(), 0.45), (StabilityAnalyzer(), 0.40), (ExposureAnalyzer(), 0.15)])
    rep = WindowPicker().pick(scorer.report(clip))
    a, b = rep.best
    assert a >= 0.8 and b <= 3.2 and rep.usable
