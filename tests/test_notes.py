import pytest

from aicut_analyzer.notes import Clip, Note, clips_of, keep_apart, pin, place, read_notes, track

FILM = """<body>
<section data-aicut-notes="1">
  <iframe id="note-1" src="notes/note.html#t=0,2" at="5" style="left: 260px; top: 40px" overrides='[{"at":"#note","text":"@这里删掉"}]'></iframe>
  <iframe id="note-1-b" src="notes/note.html#t=1.5,4" at="11.5" style="left: 260px; top: 40px"></iframe>
</section>
<section>
  <video id="a-1" src="assets/a.mov#t=10,20" at="0"></video>
  <video id="b-1" src="assets/b.mov#t=0,8" at="10" speed="2"></video>
</section>
</body>"""


class TestRead:
    def test_notes_with_their_text_and_times(self):
        # Studio splits a note as the same page with a later #t=: its length is out − in
        assert read_notes(FILM) == [Note(5.0, 2.0, "@这里删掉"), Note(11.5, 2.5, "@双击写要求")]

    def test_clips_with_their_speed(self):
        assert clips_of(FILM) == [Clip("assets/a.mov", 10, 20, 0), Clip("assets/b.mov", 0, 8, 10, 2)]

    def test_a_film_without_notes_has_none(self):
        assert read_notes("<section><video src='a.mov#t=0,1'></video></section>") == []


class TestPin:
    def test_film_time_to_the_recording_and_its_second(self):
        clips = clips_of(FILM)
        assert pin(5.0, clips) == ("assets/a.mov", 15.0)
        assert pin(11.0, clips) == ("assets/b.mov", 2.0)       # at 2x


class TestPlace:
    def test_a_note_follows_its_moment_into_the_new_cut(self):
        old = [Clip("a", 10, 20, 0)]
        new = [Clip("a", 0, 5, 0), Clip("a", 12, 20, 5)]    # the new cut drops a 5–12 s
        assert place(5.0, old, new) == pytest.approx(8.0)   # a 15 s: 5 + (15 − 12)

    def test_a_note_on_a_moment_now_cut_goes_to_the_nearest_kept_one(self):
        old = [Clip("a", 10, 20, 0)]
        new = [Clip("a", 0, 12, 0), Clip("a", 18, 20, 12)]
        assert place(5.0, old, new) == pytest.approx(12.0)  # a 15 s is gone; a 12 s is nearest

    def test_a_note_off_every_clip_stays_where_it_is(self):
        assert place(50.0, [Clip("a", 0, 10, 0)], [Clip("a", 0, 10, 0)]) == 50.0


class TestWrite:
    def test_written_notes_read_back_the_same(self):
        notes = [Note(1.0, 2.0, "@加个'音效'"), Note(4.0, 1.5, "@这里 <删掉>")]
        assert read_notes("<body>\n" + track(notes, (1080, 1920))) == notes

    def test_overlapping_notes_are_moved_apart(self):
        assert keep_apart([Note(0, 3.0, "a"), Note(2.0, 1.0, "b")]) == [Note(0, 3.0, "a"), Note(3.0, 1.0, "b")]
