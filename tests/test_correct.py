from aicut_analyzer.correct import Fix, apply_fixes, parse_fixes, sounds_alike
from aicut_analyzer.speech import Line, subtitle_lines, to_vtt
from test_speech import said


def lines_of(*texts):
    return [Line(tuple(said(*t))) for t in texts]


class TestApply:
    def test_a_fix_spanning_words_becomes_one_word_with_their_times(self):
        lines = lines_of(("做", "一个", "算", "明", "证"))
        out, done = apply_fixes(lines, [Fix(0, "算明证", "三明治")])
        w = out[0].words
        assert out[0].text == "做一个三明治" and len(done) == 1
        assert (w[-1].text, w[-1].start, w[-1].end) == ("三明治", lines[0].words[2].start, lines[0].words[4].end)

    def test_a_fix_inside_a_word_keeps_the_rest_of_it(self):
        out, _ = apply_fixes(lines_of(("刚刚", "炸了", "蓝莓")), [Fix(0, "炸", "榨")])
        assert [w.text for w in out[0].words] == ["刚刚", "榨了", "蓝莓"]

    def test_a_fix_not_in_its_line_is_not_made(self):
        lines = lines_of(("今天", "很", "好"))
        assert apply_fixes(lines, [Fix(0, "明天", "名天"), Fix(5, "很", "狠")]) == (lines, [])

    def test_a_fix_that_does_not_sound_like_what_was_heard_is_not_made(self):
        lines = lines_of(("做", "算明证"))
        assert apply_fixes(lines, [Fix(0, "算明证", "便当盒")])[1] == []

    def test_other_lines_are_left_as_they_are(self):
        lines = lines_of(("a",), ("炸", "了"))
        out, _ = apply_fixes(lines, [Fix(1, "炸", "榨")])
        assert out[0] is lines[0] and out[1].text == "榨了"

    def test_fixed_lines_still_make_timed_subtitles(self):
        out, _ = apply_fixes(subtitle_lines(said("做", "算", "明", "证", start=1.0)), [Fix(0, "算明证", "三明治")])
        assert "做<00:00:01.350>三明治" in to_vtt(out)


class TestSoundsAlike:
    def test_a_homophone_sounds_alike(self):
        assert sounds_alike("炸", "榨") and sounds_alike("算明证", "三明治")

    def test_a_guess_from_context_does_not(self):
        assert not sounds_alike("算明证", "便当盒") and not sounds_alike("花门单", "小票单")


class TestParse:
    def test_well_formed_fixes_are_read(self):
        assert parse_fixes('[{"line": 2, "wrong": "炸", "right": "榨", "why": "x"}]') == [Fix(2, "炸", "榨", "x")]

    def test_garbage_and_broken_items_are_skipped(self):
        assert parse_fixes("not json") == []
        assert parse_fixes('[{"line": "x"}, {"wrong": "a"}, {"line": 0, "wrong": "a", "right": "b"}]') == [
            Fix(0, "a", "b")]

    def test_what_trails_the_json_is_ignored(self):
        assert parse_fixes('[{"line": 0, "wrong": "a", "right": "b"}]}') == [Fix(0, "a", "b")]
        assert parse_fixes('Here you go: [{"line": 0, "wrong": "a", "right": "b"}]') == [Fix(0, "a", "b")]

    def test_a_comma_before_the_closing_bracket_is_forgiven(self):
        assert parse_fixes('[{"line": 0, "wrong": "a", "right": "b"},\n]') == [Fix(0, "a", "b")]
