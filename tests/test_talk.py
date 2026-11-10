import numpy as np
import pytest

from aicut_analyzer.speech import Word
from datetime import datetime, timedelta
from pathlib import Path

from aicut_analyzer.talk import (RATE, Part, Probe, Shot, TalkCut, cut_talk, ducked_music, film_html,
                                 in_day_order, in_shooting_order, judged_shows, reel_speech, scenes, shot_time, silences, subtract,
                                 with_shows, without_adjusting, gain, level, level_to, measure)
from aicut_analyzer.speech import write_wav


def probe_seconds(path):
    import subprocess
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True).stdout)
from test_speech import said


def cut_of(ranges):
    return TalkCut(ranges, [], [])


class TestCutTalk:
    def test_the_kept_words_are_the_ones_inside_the_ranges(self):
        w = said("我们", "我们", "去", "嗯", "爬山", start=0.5)
        cut = cut_talk(w, 5.0)
        assert [x.text for x in cut.kept_words] == ["我们", "去", "爬山"]
        assert {r.why for r in cut.removed} >= {"repeat", "filler"}

    def test_a_recording_nobody_speaks_in_is_all_one_silent_stretch(self):
        cut = cut_talk([], 30.0)
        assert cut.ranges == [] and [(r.start, r.end, r.why) for r in cut.removed] == [(0, 30.0, "pause")]
        assert with_shows(cut, [(10.0, 18.0)], longest=4.0).ranges == [(10.0, 18.0)]

    def test_film_spans_follow_one_another(self):
        cut = cut_of([(1.0, 2.5), (4.0, 4.5), (6.0, 8.0)])
        assert cut.film_spans() == [(0, 1.5), (1.5, 2.0), (2.0, 4.0)]
        assert cut.length == pytest.approx(4.0)


class TestFilm:
    def test_a_clip_per_range_placed_end_to_end(self):
        html = film_html([Part("assets/a b.MOV", "a b", cut_of([(1.0, 2.5), (4.0, 4.5)]))], (1080, 1920))
        assert 'content="width=1080, height=1920"' in html
        assert '<video id="a-b-1" src="assets/a b.MOV#t=1,2.5" at="0"></video>' in html
        assert '<video id="a-b-2" src="assets/a b.MOV#t=4,4.5" at="1.5"></video>' in html
        assert "<audio" not in html

    def test_music_goes_on_its_own_track_as_long_as_the_film(self):
        html = film_html([Part("assets/a.mp4", "a", cut_of([(0, 3.0)]))], (1920, 1080), "assets/bed-ducked.wav", 0.4)
        assert html.count("<section>") == 2
        assert '<audio id="music" src="assets/bed-ducked.wav#t=0,3.000" volume="0.4"></audio>' in html


class TestDuckedMusic:
    MUSIC = np.ones((RATE, 2), np.float32)     # 1 s, looped

    def test_music_is_as_long_as_the_film_and_lowered_under_speech(self):
        w = [Word("a", 0.5, 1.5)]
        out = ducked_music(self.MUSIC, 4.0, TalkCut([(0, 4.0)], [], w).film_speech(), level=0.25, fade_out=0.5)
        assert len(out) == 4 * RATE
        assert out[RATE, 0] == pytest.approx(0.25)
        assert out[-1, 0] == pytest.approx(0, abs=1e-3)

    def test_it_comes_back_up_where_a_kept_stretch_has_no_words(self):
        w = [Word("a", 0.2, 1.0), Word("b", 20.2, 21.0)]
        cut = TalkCut([(0, 8.0), (20.0, 21.2)], [], w)       # b plays at 8.2 s in the film
        out = ducked_music(self.MUSIC, cut.length, cut.film_speech(), level=0.25, fade_out=0.1)
        assert out[int(4.0 * RATE), 0] == pytest.approx(1.0)
        assert out[int(8.5 * RATE), 0] == pytest.approx(0.25)


class TestShows:
    def test_a_show_is_put_back_sped_up_between_the_kept_ranges(self):
        cut = with_shows(cut_of([(0, 2.0), (12.0, 14.0)]), [(4.0, 10.0)], longest=2.0)
        assert cut.ranges == [(0, 2.0), (4.0, 10.0), (12.0, 14.0)]
        assert [cut.speed(i) for i in range(3)] == [1, 3.0, 1]
        assert cut.film_spans() == [(0, 2.0), (2.0, 4.0), (4.0, 6.0)]

    def test_speed_stays_between_its_limits(self):
        cut = with_shows(cut_of([(0, 1.0)]), [(2.0, 3.0), (5.0, 45.0)], longest=4.0)
        assert [cut.speed(1), cut.speed(2)] == [2.0, 4.0]

    def test_a_show_over_kept_speech_is_refused(self):
        with pytest.raises(ValueError):
            with_shows(cut_of([(0, 5.0)]), [(4.0, 8.0)])

    def test_a_sped_up_clip_says_its_speed(self):
        html = film_html([Part("a.mp4", "a", with_shows(cut_of([(0, 1.0)]), [(2.0, 10.0)], longest=2.0))], (1080, 1920))
        assert '<video id="a-2" src="a.mp4#t=2,10" at="1" speed="4"></video>' in html
        assert 'id="a-1" src="a.mp4#t=0,1" at="0"></video>' in html

    def test_words_after_a_sped_up_stretch_land_where_they_play(self):
        w = [Word("a", 0.1, 0.9), Word("b", 10.2, 10.8)]
        cut = with_shows(TalkCut([(0, 1.0), (10.0, 11.0)], [], w), [(2.0, 6.0)], longest=2.0)
        assert cut.film_speech()[1] == pytest.approx((3.2, 3.8))


T0 = datetime.fromisoformat("2026-11-02T13:11:35-08:00")


def shot(name, minutes, duration=10.0, timed=True):
    return Shot(Path(name), Probe(duration, 1080, 1920, T0 + timedelta(minutes=minutes) if timed else None))


class TestOrder:
    def test_an_iphone_creation_date_keeps_its_time_zone(self):
        t = shot_time({"creation_time": "2026-11-02T21:11:35.000000Z",
                       "com.apple.quicktime.creationdate": "2026-11-02T13:11:35-0800"})
        assert t == T0 and t.utcoffset() == timedelta(hours=-8)

    def test_without_the_iphone_date_the_container_time_is_used(self):
        assert shot_time({"creation_time": "2026-11-02T21:11:35.000000Z"}) == T0
        assert shot_time({}) is None

    def test_shots_go_by_when_they_were_shot_not_by_name(self):
        shots = [shot("IMG_0002.MOV", 5), shot("1.MOV", 0), shot("IMG_0003.MOV", 2)]
        assert [s.path.name for s in in_shooting_order(shots)] == ["1.MOV", "IMG_0003.MOV", "IMG_0002.MOV"]

    def test_shots_without_a_time_come_last(self):
        shots = [shot("b.mov", 0, timed=False), shot("a.mov", 3)]
        assert [s.path.name for s in in_shooting_order(shots)] == ["a.mov", "b.mov"]

    def test_a_long_gap_starts_a_scene(self):
        shots = [shot("a", 0, 60), shot("b", 2), shot("c", 30), shot("d", 31)]
        assert [[s.path.name for s in g] for g in scenes(shots, gap=600)] == [["a", "b"], ["c", "d"]]

    def test_the_gap_is_from_the_end_of_a_long_shot(self):
        # a runs 15 minutes: b starts 1 minute after it ends, so it is the same scene
        assert len(scenes([shot("a", 0, 900), shot("b", 16)], gap=600)) == 1


class TestReel:
    def test_parts_play_one_after_another(self):
        parts = [Part("assets/a.mov", "a", cut_of([(0, 2.0)])), Part("assets/b.mov", "b", cut_of([(5.0, 8.0)]))]
        html = film_html(parts, (1080, 1920))
        assert '<video id="b-1" src="assets/b.mov#t=5,8" at="2"></video>' in html

    def test_speech_in_a_later_part_is_placed_after_the_earlier_ones(self):
        parts = [Part("a", "a", cut_of([(0, 2.0)])),
                 Part("b", "b", TalkCut([(1.0, 3.0)], [], [Word("x", 1.5, 2.0)]))]
        assert reel_speech(parts) == [pytest.approx((2.5, 3.0))]


class Seg:
    def __init__(self, start, end, kind, best=None, what=""):
        self.start, self.end, self.kind, self.best, self.what = start, end, kind, best, what


class TestSubtract:
    def test_a_hole_is_taken_out_of_a_range(self):
        assert subtract([(0, 10.0)], [(3.0, 5.0)], []) == [(0, 3.0), (5.0, 10.0)]

    def test_a_hole_stops_at_words(self):
        assert subtract([(0, 10.0)], [(3.0, 6.0)], [Word("a", 4.0, 4.5)]) == [(0, 3.0), (4.0, 4.5), (6.0, 10.0)]

    def test_a_hole_never_cuts_between_words(self):
        w = [Word("a", 3.0, 3.4), Word("b", 3.42, 3.8)]
        assert subtract([(2.0, 6.0)], [(2.5, 4.0)], w) == [(2.0, 2.5), (3.0, 6.0)]

    def test_slivers_are_dropped(self):
        assert subtract([(0, 3.1)], [(0, 3.0)], []) == []


class TestAdjusting:
    def test_camera_moves_are_cut_from_a_b_roll_clip(self):
        cut = without_adjusting(cut_of([(0, 4.0)]), [Seg(0, 1.2, "adjust"), Seg(1.2, 4.0, "show")])
        assert cut.ranges == [(1.2, 4.0)] and cut.removed[0].why == "adjust"

    def test_speech_over_a_camera_move_is_kept(self):
        w = [Word("a", 1.0, 1.4)]
        cut = without_adjusting(TalkCut([(0.8, 2.0)], [], w), [Seg(0.5, 1.8, "adjust")])
        assert cut.ranges == [(1.0, 1.4)]


class TestSilences:
    def test_what_the_cut_leaves_out_where_nobody_speaks(self):
        w = [Word("a", 1.0, 2.0), Word("嗯", 5.0, 5.3), Word("b", 12.0, 13.0)]
        # the cut keeps a and b; between them a filler at 5 s, which is not silence
        out = silences([(0.9, 2.2), (11.9, 13.2)], w, 20.0, shortest=2.0, pad=0.1)
        assert out == [(2.2, 4.9), (5.4, 11.9), (13.2, 20.0)]

    def test_short_gaps_are_not_silences(self):
        assert silences([(0, 5.0), (6.0, 10.0)], [], 10.0) == []


class TestEnds:
    def test_a_short_shot_loses_camera_moves_only_at_its_ends(self):
        segs = [Seg(0, 0.5, "adjust"), Seg(1.0, 1.5, "adjust"), Seg(2.0, 2.6, "adjust")]
        cut = without_adjusting(cut_of([(0, 2.6)]), segs, ends_only=2.6)
        assert cut.ranges == [(0.5, 2.0)]


class TestJudgedShows:
    def test_only_the_best_seconds_of_a_show_are_kept(self):
        segs = [Seg(10, 20, "show", best=(12.0, 14.5)), Seg(20, 30, "filler")]
        assert judged_shows([(9.0, 31.0)], segs) == [(12.0, 14.5)]

    def test_a_long_best_is_cut_to_its_middle(self):
        assert judged_shows([(0, 60.0)], [Seg(37, 53, "show", best=(37.5, 52.0))], longest=3.0) == [(43.25, 46.25)]

    def test_filler_and_camera_moves_are_left_out(self):
        assert judged_shows([(0, 30.0)], [Seg(0, 10, "filler"), Seg(10, 12, "adjust")]) == []

    def test_a_show_is_kept_inside_the_silence(self):
        # the speaker talks again at 15: the show may not run into it
        assert judged_shows([(10.0, 15.0)], [Seg(12, 20, "show", best=(13.0, 16.0))]) == [(13.0, 15.0)]


class TestLevel:
    def test_quiet_speech_is_raised_to_the_film_level(self):
        assert gain(-38.0, speech=True) == 22.0

    def test_loud_speech_is_lowered(self):
        assert gain(-10.0, speech=True) == -6.0

    def test_a_clip_without_speech_sits_lower_and_is_raised_little(self):
        assert gain(-23.0, speech=False) == 0.0
        assert gain(-45.0, speech=False) == 10.0

    def test_silence_is_left_alone(self):
        assert gain(float("-inf"), speech=True) == 0.0

    def test_a_levelled_part_plays_its_own_sound_in_step_with_the_picture(self):
        cut = TalkCut([(3.0, 5.0), (8.0, 9.0)], [], [Word("a", 3.5, 4.0)])
        html = film_html([Part("a.mov", "a", cut, "a.speech.wav", "a.ambient.wav")], (1080, 1920))
        assert '<video id="a-1" src="a.mov#t=3,5" at="0" volume="0"></video>' in html
        assert '<audio id="a-1-sound" src="a.speech.wav#t=3,5" at="0"></audio>' in html
        assert '<audio id="a-2-sound" src="a.ambient.wav#t=8,9" at="2"></audio>' in html

    def test_levelling_reaches_the_level_and_keeps_peaks_under_the_ceiling(self, tmp_path):
        rng = np.random.default_rng(0)
        t = np.arange(4 * RATE) / RATE
        voice = (0.02 * np.sin(2 * np.pi * 220 * t))[:, None].repeat(2, axis=1).astype(np.float32)
        voice[RATE:RATE + 200] = 0.99           # a clatter at full scale
        write_wav(tmp_path / "x.wav", voice)
        _, after, peak = level_to(tmp_path / "x.wav", tmp_path / "y.wav", [(0, 4.0)], speech=True, length=4.5)
        assert after == pytest.approx(-16.0, abs=1.5) and peak <= -0.9

    def test_measured_loudness_follows_the_sound(self, tmp_path):
        t = np.arange(3 * RATE) / RATE
        tone = np.sin(2 * np.pi * 440 * t)[:, None].repeat(2, axis=1).astype(np.float32)
        write_wav(tmp_path / "quiet.wav", tone * 0.1)
        write_wav(tmp_path / "loud.wav", tone * 0.2)
        q, _ = measure(tmp_path / "quiet.wav", [(0, 3.0)])
        l, _ = measure(tmp_path / "loud.wav", [(0, 3.0)])
        assert l - q == pytest.approx(6.0, abs=0.3)

    def test_only_the_kept_stretches_are_measured(self, tmp_path):
        t = np.arange(4 * RATE) / RATE
        tone = np.sin(2 * np.pi * 440 * t)[:, None].repeat(2, axis=1).astype(np.float32)
        tone[2 * RATE:] *= 0.1          # quiet second half
        write_wav(tmp_path / "x.wav", tone * 0.2)
        first, _ = measure(tmp_path / "x.wav", [(0, 2.0)])
        second, _ = measure(tmp_path / "x.wav", [(2.0, 4.0)])
        assert first - second == pytest.approx(20.0, abs=0.5)

    def test_the_levelled_sound_is_as_long_as_the_video(self, tmp_path):
        write_wav(tmp_path / "x.wav", np.zeros((RATE, 2), np.float32) + 0.01)
        level(tmp_path / "x.wav", tmp_path / "y.wav", 0.0, 1.25)
        assert probe_seconds(tmp_path / "y.wav") >= 1.25


class TestDays:
    def at(self, name, when):
        return Shot(Path(name), Probe(10.0, 1080, 1920, datetime.fromisoformat(when)))

    def test_days_go_in_the_order_given_each_in_shooting_order(self):
        shots = [self.at("seagull", "2026-10-30T14:09:16-07:00"), self.at("ghost", "2026-11-01T18:01:47-08:00"),
                 self.at("sandwich-2", "2026-11-02T13:20:55-08:00"), self.at("sandwich-1", "2026-11-02T13:11:35-08:00")]
        out = in_day_order(shots, ["11-02", "10-30", "11-01"])
        assert [s.path.name for s in out] == ["sandwich-1", "sandwich-2", "seagull", "ghost"]

    def test_days_not_given_follow_in_shooting_order(self):
        shots = [self.at("a", "2026-10-30T14:00:00-07:00"), self.at("b", "2026-11-02T13:00:00-08:00")]
        assert [s.path.name for s in in_day_order(shots, ["11-02"])] == ["b", "a"]

    def test_going_back_a_day_starts_a_scene(self):
        shots = [self.at("b", "2026-11-02T13:00:00-08:00"), self.at("a", "2026-10-30T14:00:00-07:00")]
        assert len(scenes(shots)) == 2
