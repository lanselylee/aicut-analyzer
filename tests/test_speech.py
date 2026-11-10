import numpy as np
import pytest

from aicut_analyzer.speech import (Word, cjk_punct, sentence_words, duck_gain, keep_ranges, plausible, removed, repeated, retakes, snap_to_quiet,
                                   subtitle_lines, tighten, to_vtt)


def said(*parts, start=1.0, gap=0.05, length=0.3):
    """Words one after another; a number in `parts` is a pause of that many seconds instead of the usual gap."""
    words, t = [], start
    for p in parts:
        if isinstance(p, (int, float)):
            t += p - gap
            continue
        words.append(Word(p, round(t, 3), round(t + length, 3)))
        t += length + gap
    return words


def texts(words, idx):
    return [words[i].text for i in sorted(idx)]


class TestRepeated:
    def test_a_restarted_phrase_drops_the_first_try(self):
        w = said("我们", "今天", "我们", "今天", "去", "爬山")
        assert sorted(repeated(w)) == [0, 1]

    def test_a_cut_short_word_still_counts_as_the_same_try(self):
        w = said("我们", "今", "我们", "今天", "去")
        assert texts(w, repeated(w)) == ["我们", "今"]

    def test_a_stutter_with_a_pause_is_dropped(self):
        w = said("我", 0.3, "我", "觉得")
        assert texts(w, repeated(w)) == ["我"] and 0 in repeated(w)

    def test_a_doubled_word_said_in_one_breath_is_a_word(self):
        # 谢谢, 看看: the same character twice is often one word
        assert repeated(said("谢", "谢", "大家", gap=0.0)) == set()

    def test_fillers_between_the_tries_are_looked_through(self):
        w = said("我们", "去", "嗯", "我们", "去", "爬山")
        assert sorted(repeated(w)) == [0, 1]

    def test_a_phrase_said_again_much_later_is_not_a_repeat(self):
        assert repeated(said("好", "的", 3.0, "好", "的")) == set()

    def test_punctuation_and_case_do_not_hide_a_repeat(self):
        assert sorted(repeated(said("So,", "we", "so", "we", "went"))) == [0, 1]

    def test_different_words_are_never_dropped(self):
        assert repeated(said("今天", "天气", "很", "好")) == set()


class TestRetakes:
    def test_a_sentence_said_again_keeps_the_last_take(self):
        w = said("哈喽", "大家，", "中午", "好。", 3.5, "哈喽", "大家，", "中午", "好。", "我", "刚刚")
        assert sorted(retakes(w)) == [0, 1, 2, 3]

    def test_a_take_that_differs_a_little_still_counts(self):
        w = said("因为", "今天", "我们", "是", "一天", "的", "课", 0.8, "今天", "是", "一天", "的", "课")
        assert sorted(retakes(w)) == list(range(7))

    def test_a_first_take_that_paused_in_the_middle_is_one_take(self):
        w = said("哈", "喽", "大家，", 0.7, "中", "午", "好。", 3.5, "哈", "喽", "大家，", "中", "午", "好。")
        assert sorted(retakes(w)) == list(range(6))

    def test_what_was_given_up_between_the_takes_goes_with_the_first(self):
        w = said("因为", "今天", "我们", "是", "一天", "的", "课", 0.7, "但是", "我们", "去", 0.7,
                 "今天", "是", "一天", "的", "课", "他", "中午", "没", "吃饭")
        assert texts(w, retakes(w)) == ["因为", "今天", "我们", "是", "一天", "的", "课", "但是", "我们", "去"]

    def test_the_retake_may_go_on_past_the_first_take(self):
        w = said("中午", "好。", 2.0, "中午", "好，", "我", "刚刚", "做", "了", "饭")
        assert sorted(retakes(w)) == [0, 1]

    def test_lines_built_alike_are_not_takes(self):
        w = said("我", "很", "喜欢", "这个。", "我", "很", "喜欢", "那个。")
        assert retakes(w) == set()

    def test_a_long_stretch_between_means_the_speaker_moved_on(self):
        w = said("今天", "是", "一天", "的", "课。", "然后", "我们", "去", "了", "超市", "买", "了", "很多", "东西",
                 "回来", "以后", "今天", "是", "一天", "的", "课。")
        assert retakes(w) == set()

    def test_different_sentences_are_not_takes(self):
        w = said("今天", "去", "爬山。", "明天", "去", "游泳。")
        assert retakes(w) == set()

    def test_a_sentence_said_again_much_later_is_kept(self):
        w = said("中午", "好。", 20.0, "中午", "好。")
        assert retakes(w) == set()

    def test_short_answers_are_never_takes(self):
        assert retakes(said("好。", 1.0, "好。")) == set()


class TestKeepRanges:
    def test_continuous_speech_is_one_range_with_room_around_it(self):
        w = said("今天", "天气", "很", "好")
        assert keep_ranges(w, 0, 10) == [(pytest.approx(w[0].start - 0.12), pytest.approx(w[-1].end + 0.18))]

    def test_a_long_pause_is_cut_down(self):
        w = said("第一", "句", 2.0, "第二", "句")
        (a1, b1), (a2, b2) = keep_ranges(w, 0, 10)
        assert b1 == pytest.approx(w[1].end + 0.18) and a2 == pytest.approx(w[2].start - 0.12)

    def test_a_short_pause_stays(self):
        assert len(keep_ranges(said("第一", "句", 0.3, "第二", "句"), 0, 10)) == 1

    def test_a_filler_is_cut_out_even_without_a_pause(self):
        w = said("我", "嗯", "觉得")
        (a1, b1), (a2, b2) = keep_ranges(w, 0, 10)
        assert b1 <= w[1].start and a2 >= w[1].end

    def test_the_first_try_of_a_restart_is_cut_out(self):
        w = said("我们", "今天", "我们", "今天", "去")
        ranges = keep_ranges(w, 0, 10)
        kept = [x for x in w if any(a <= x.start and x.end <= b for a, b in ranges)]
        assert [x.text for x in kept] == ["我们", "今天", "去"] and kept[0] is w[2]

    def test_keeping_fillers_and_repeats_when_asked(self):
        w = said("我", "嗯", "我们", "我们", "去")
        assert len(keep_ranges(w, 0, 10, drop_fillers=False, drop_repeats=False)) == 1

    def test_no_edge_is_ever_inside_a_word(self):
        w = said("a", 0.6, "b", "嗯", "c", "c", "d", 1.2, "e", start=0.05)
        for edge in (t for r in keep_ranges(w, 0, 10) for t in r):
            assert not any(x.start < edge < x.end for x in w)

    def test_ranges_stay_inside_the_stretch(self):
        w = said("a", "b", start=0.02)
        (a, b), = keep_ranges(w, 0, w[-1].end + 0.05)
        assert a == 0 and b == pytest.approx(w[-1].end + 0.05)

    def test_without_words_everything_is_kept(self):
        assert keep_ranges([], 2, 5) == [(2, 5)]


class TestRemoved:
    def test_each_cut_is_named_by_what_was_in_it(self):
        w = said("开始", 2.0, "好", "我们", "我们", "去", "嗯", "吧", start=0.12)
        r = removed(w, 0, 10, keep_ranges(w, 0, 10))
        assert [x.why for x in r] == ["pause", "repeat", "filler", "pause"]  # the last: after the final word
        assert r[1].text == "我们"

    def test_a_retake_is_named(self):
        w = said("中午", "好。", 2.0, "中午", "好。", start=0.12)
        assert [x.why for x in removed(w, 0, 10, keep_ranges(w, 0, 10))][:1] == ["retake"]


class TestSnap:
    RATE = 100

    def test_an_edge_moves_to_the_quiet_moment_nearby(self):
        loud = np.ones(500)
        loud[205] = 0.0
        assert snap_to_quiet([(1.0, 2.0)], [], loud, self.RATE) == [(1.0, 2.05)]

    def test_an_edge_never_moves_into_a_word(self):
        loud = np.ones(500)
        loud[300] = 0.0       # quiet, but inside a word (a stop consonant)
        out = snap_to_quiet([(1.0, 2.95)], [Word("x", 2.97, 3.2)], loud, self.RATE)
        assert out[0][1] <= 2.97

    def test_an_edge_does_not_move_farther_than_reach(self):
        loud = np.ones(500)
        loud[250] = 0.0
        assert snap_to_quiet([(1.0, 2.0)], [], loud, self.RATE) == [(1.0, 2.0)]


class TestTighten:
    RATE = 100

    def loud(self, *spans):
        x = np.full(1000, 0.001)
        for a, b in spans:
            x[int(a * self.RATE):int(b * self.RATE)] = 0.5
        return x

    def test_silence_whisper_gave_a_word_is_taken_off(self):
        # sound only from 9.8 s, but the word is marked from 9.38 s
        out = tighten([Word("这", 9.38, 9.94)], self.loud((9.8, 9.94)), self.RATE)
        assert out[0].start == pytest.approx(9.8) and out[0].end == pytest.approx(9.94)

    def test_a_word_that_is_all_sound_is_left_alone(self):
        w = [Word("好", 1.0, 1.5)]
        assert tighten(w, self.loud((0.9, 1.6)), self.RATE) == w

    def test_a_word_is_never_left_too_short(self):
        out = tighten([Word("x", 2.0, 3.0)], self.loud((2.5, 2.51)), self.RATE, shortest=0.08)
        assert out[0].end - out[0].start >= 0.08 - 1e-9

    def test_a_word_is_only_ever_made_shorter(self):
        out = tighten([Word("a", 1.0, 1.2)], self.loud((0.5, 2.0)), self.RATE)
        assert (out[0].start, out[0].end) == (1.0, 1.2)


class TestPunct:
    def test_commas_after_chinese_become_full_width(self):
        assert cjk_punct("大家好,") == "大家好，"

    def test_english_keeps_its_commas(self):
        assert cjk_punct("so,") == "so,"


class TestSubtitles:
    def test_a_sentence_end_starts_a_new_line(self):
        lines = subtitle_lines(said("今天", "很", "好。", "我们", "走"))
        assert [l.text for l in lines] == ["今天很好。", "我们走"]

    def test_a_long_line_breaks_after_its_comma(self):
        w = said("今天天气", "非常好，", "我们", "一起", "去", "公园", "散步", "然后", "回家")
        lines = subtitle_lines(w, max_chars=14)
        assert lines[0].text == "今天天气非常好，"

    def test_a_pause_starts_a_new_line(self):
        assert len(subtitle_lines(said("a", 1.0, "b"))) == 2

    def test_a_full_width_comma_needs_no_space(self):
        assert subtitle_lines(said("大家好，", "今天", gap=0.1))[0].text == "大家好，今天"

    def test_fillers_are_left_out(self):
        assert [l.text for l in subtitle_lines(said("我", "嗯", "去"))] == ["我去"]

    def test_english_words_keep_their_spaces(self):
        assert subtitle_lines(said("we", "went", "home"))[0].text == "we went home"

    def test_vtt_times_each_word(self):
        vtt = to_vtt(subtitle_lines(said("we", "went", start=1.0)))
        assert vtt.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:01.650\nwe <00:00:01.350>went" in vtt


class TestDuck:
    def test_music_is_down_while_speaking_and_up_away_from_it(self):
        t = np.array([0.0, 5.0, 5.5, 6.0, 10.0])
        g = duck_gain(t, [(5.0, 6.0)], level=0.25)
        assert g[0] == 1 and g[-1] == 1 and np.allclose(g[1:4], 0.25)

    def test_it_starts_down_before_the_first_word(self):
        g = duck_gain(np.array([4.8]), [(5.0, 6.0)], attack=0.25)
        assert 0.25 < g[0] < 1

    def test_it_stays_down_across_a_short_pause(self):
        g = duck_gain(np.array([6.5]), [(5.0, 6.0), (7.0, 8.0)], level=0.25, bridge=1.2)
        assert g[0] == pytest.approx(0.25)


class TestPlausible:
    def test_stock_subtitle_phrases_are_dropped(self):
        assert not plausible({"text": "字幕志愿者 李宗盛"})
        assert not plausible({"text": " Thank you for watching!"})

    def test_a_looping_segment_is_dropped(self):
        assert not plausible({"text": "好的好的好的", "compression_ratio": 3.1})

    def test_a_guess_over_noise_is_dropped(self):
        # over cooking or music: low log-probability although no_speech_prob stays 0
        assert not plausible({"text": "macbook", "avg_logprob": -4.55, "no_speech_prob": 0.0})

    def test_one_syllable_over_and_over_is_dropped(self):
        assert not plausible({"text": "喇喇喇喇喇喇"})

    def test_song_credits_are_dropped(self):
        assert not plausible({"text": "作词・作曲・编曲 初音ミク"})

    def test_ordinary_speech_is_kept(self):
        assert plausible({"text": "今天我们去爬山", "compression_ratio": 1.1, "words": [{"probability": 0.9}]})


class TestParaformerWords:
    def test_a_character_per_timestamp_with_its_punctuation(self):
        w = sentence_words({"text": "份鸡汤。", "start": 96000, "end": 96715,
                            "timestamp": [[96000, 96180], [96180, 96380], [96380, 96715]]})
        assert [(x.text, x.start, x.end) for x in w] == [("份", 96.0, 96.18), ("鸡", 96.18, 96.38), ("汤。", 96.38, 96.715)]

    def test_a_latin_word_is_one_unit(self):
        w = sentence_words({"text": " hello，大家", "start": 0, "end": 900,
                            "timestamp": [[0, 400], [400, 600], [600, 900]]})
        assert [x.text for x in w] == ["hello，", "大", "家"]

    def test_when_counts_differ_the_span_is_shared(self):
        w = sentence_words({"text": "好的", "start": 1000, "end": 2000, "timestamp": [[1000, 2000]]})
        assert [(x.start, x.end) for x in w] == [(1.0, 1.5), (1.5, 2.0)]
