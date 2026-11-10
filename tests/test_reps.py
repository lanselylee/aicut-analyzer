import numpy as np
import pytest

from aicut_analyzer.reps import RepAnalyzer, SetFinder, periodicity
from synthetic import FPS, as_clip, workout


def analyze(segments, **kw):
    frames, ends = workout(segments, **kw)
    return RepAnalyzer().analyze(as_clip(frames)), ends


class WholeClip(SetFinder):
    """A set finder that calls the whole clip one stretch, to test the rep counter on its own."""

    def __init__(self, period):
        self.period = period

    def find(self, x, fps):
        return [(0, len(x), self.period)]


class TestPeriodicity:
    def test_a_sine_repeats_at_its_period(self):
        r, lag = periodicity(np.sin(2 * np.pi * np.arange(150) / 30), 8, 75)
        assert r >= 0.75 and lag == 30   # a 150-sample sine overlaps itself on 120 samples at one period

    def test_a_slow_ramp_does_not_count_as_repeating(self):
        # correlates with itself at short lags only because it changes slowly
        assert periodicity(np.linspace(0, 1, 150) ** 2, 8, 75)[0] == 0.0

    def test_noise_on_a_slow_drift_does_not_count_as_repeating(self):
        # noise makes small bumps in the autocorrelation, but the drift keeps it above zero: nothing came back
        s = np.linspace(0, 1, 150) + np.random.default_rng(0).normal(0, 0.03, 150)
        assert periodicity(s, 8, 75)[0] == 0.0


class TestSets:
    def test_one_set_with_every_rep_counted(self):
        sets, ends = analyze([("still", 4), ("reps", 8), ("still", 4)])
        assert len(sets) == 1
        assert len(sets[0].reps) == 8

    def test_the_set_spans_the_reps_not_the_rest_around_them(self):
        sets, ends = analyze([("still", 4), ("reps", 8), ("still", 4)])
        assert sets[0].start == pytest.approx(4.0, abs=0.2)
        assert sets[0].end == pytest.approx(ends[-1], abs=0.2)

    def test_two_sets_with_a_rest_between_are_two(self):
        sets, _ = analyze([("still", 3), ("reps", 6), ("still", 12), ("reps", 6), ("still", 3)])
        assert [len(s.reps) for s in sets] == [6, 6]

    def test_still_footage_has_no_set(self):
        assert analyze([("still", 20)])[0] == []

    def test_two_movements_alike_are_not_a_set(self):
        frames, _ = workout([("still", 5), ("reps", 2), ("still", 5)])
        assert RepAnalyzer(finder=WholeClip(period=2.0)).analyze(as_clip(frames)) == []

    def test_fidgeting_is_not_a_set(self):
        assert analyze([("still", 2), ("fidget", 16), ("still", 2)])[0] == []

    def test_fidgeting_before_the_set_is_left_out(self):
        sets, ends = analyze([("fidget", 8), ("still", 2), ("reps", 8), ("still", 3)])
        assert len(sets) == 1 and len(sets[0].reps) == 8
        assert sets[0].start == pytest.approx(10.0, abs=0.3)

    def test_fidgeting_right_after_the_last_rep_is_left_out(self):
        # putting the bar back: movement that goes on without a pause, but at no steady pace or size
        sets, ends = analyze([("still", 3), ("reps", 6), ("fidget", 6), ("still", 3)])
        assert len(sets) == 1 and len(sets[0].reps) == 6
        assert sets[0].end == pytest.approx(ends[-1], abs=0.2)

    def test_slow_light_change_does_not_break_the_count(self):
        sets, _ = analyze([("still", 3), ("reps", 8), ("still", 3)], light_drift=60)
        assert len(sets[0].reps) == 8

    @pytest.mark.parametrize("period", [1.2, 2.0, 3.5])
    def test_fast_and_slow_movements(self, period):
        sets, _ = analyze([("still", 3), ("reps", 7), ("still", 3)], period=period)
        assert len(sets) == 1 and len(sets[0].reps) == 7
        assert sets[0].period == pytest.approx(period, abs=0.15)


class TestTurns:
    def test_reps_end_when_the_bar_is_back_at_rest(self):
        sets, ends = analyze([("still", 4), ("reps", 8), ("still", 4)])
        got = [r.end for r in sets[0].reps]
        assert np.abs(np.array(got) - np.array(ends)).max() <= 2 / FPS   # two sampled frames

    def test_the_turn_is_halfway_through_the_rep(self):
        sets, _ = analyze([("still", 4), ("reps", 8), ("still", 4)], period=2.0)
        for r in sets[0].reps:
            assert r.turn - r.start == pytest.approx(1.0, abs=2 / FPS)

    def test_the_movement_is_still_at_its_turns_and_fast_between_them(self):
        ws = analyze([("still", 4), ("reps", 8), ("still", 4)])[0][0]
        assert max(ws.still_at(t) for t in ws.turns()) < 0.15
        turns = ws.turns()
        for a, b in zip(turns, turns[1:]):
            between = (ws.times > a) & (ws.times < b)
            assert ws.speed[between].max() > 0.7


class TestLikeness:
    def test_reps_of_one_set_are_alike(self):
        ws = analyze([("still", 4), ("reps", 8), ("still", 4)])[0][0]
        assert min(r.likeness for r in ws.reps) > 0.95
