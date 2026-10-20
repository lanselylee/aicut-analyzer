import numpy as np

from aicut_analyzer.analyzers import ExposureAnalyzer, SharpnessAnalyzer, StabilityAnalyzer
from synthetic import as_clip, footage


def scores(analyzer, frames):
    return analyzer.score(analyzer.raw(as_clip(frames)))


class TestSharpness:
    def test_blurred_frames_score_low_and_the_rest_high(self):
        s = scores(SharpnessAnalyzer(), footage(60, pan=(1, 2), blur=slice(25, 35)))
        assert s[25:35].max() < 0.4
        assert s[:20].min() > 0.8 and s[40:].min() > 0.8

    def test_a_scene_with_less_detail_is_not_mistaken_for_blur(self):
        # the first half shows a busy scene, the second a plain one (both in focus): judged against neighbours,
        # the plain half stays sharp
        busy = footage(30, seed=0)
        plain = [np.clip(f.astype(np.int16) // 4 + 96, 0, 255).astype(np.uint8) for f in footage(30, seed=3)]
        s = scores(SharpnessAnalyzer(), busy + plain)
        assert s[45:].min() > 0.8

    def test_judging_against_the_whole_clip_would_get_that_wrong(self):
        busy = footage(30, seed=0)
        plain = [np.clip(f.astype(np.int16) // 4 + 96, 0, 255).astype(np.uint8) for f in footage(30, seed=3)]
        s = scores(SharpnessAnalyzer(reach=10_000), busy + plain)
        assert s[45:].max() < 0.5


class TestStability:
    def test_a_still_shot_is_stable(self):
        assert scores(StabilityAnalyzer(), footage(45)).mean() > 0.95

    def test_a_steady_pan_is_not_shake(self):
        assert scores(StabilityAnalyzer(), footage(45, pan=(3, 2))).mean() > 0.85

    def test_hand_shake_scores_low(self):
        steady = scores(StabilityAnalyzer(), footage(45, pan=(3, 2))).mean()
        shaky = scores(StabilityAnalyzer(), footage(45, pan=(3, 2), shake=6)).mean()
        assert shaky < 0.4 < steady

    def test_more_shake_scores_lower(self):
        means = [scores(StabilityAnalyzer(), footage(45, shake=s)).mean() for s in (1, 3, 6)]
        assert means[0] > means[1] > means[2]


class TestExposure:
    def test_normal_frames_score_high(self):
        assert scores(ExposureAnalyzer(), footage(20)).min() > 0.9

    def test_blown_highlights_score_low(self):
        s = scores(ExposureAnalyzer(), footage(30, clip_white=slice(10, 20)))
        assert s[10:20].max() < 0.3
        assert s[:10].min() > 0.9 and s[20:].min() > 0.9
