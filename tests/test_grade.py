import numpy as np
import pytest

from aicut_analyzer.grade import (Look, Shift, band, grade, lut, oklab_to_srgb, portra400, srgb_to_oklab, tone,
                                  vision500t, write_cube)


def hue_chroma(rgb):
    lab = srgb_to_oklab(np.array(rgb, float))
    return np.degrees(np.arctan2(lab[2], lab[1])) % 360, np.hypot(lab[1], lab[2])


class TestSpaces:
    def test_oklab_round_trips(self):
        rgb = np.random.default_rng(0).random((100, 3))
        assert np.allclose(oklab_to_srgb(srgb_to_oklab(rgb)), rgb, atol=1e-6)


class TestTone:
    def test_blacks_are_lifted_and_white_stays_white(self):
        look = Look("x", black=0.05, shoulder=0.7)
        assert tone(np.array(0.0), look) == pytest.approx(0.05)
        assert tone(np.array(1.0), look) == pytest.approx(1.0)

    def test_highlights_roll_off_instead_of_clipping(self):
        look = Look("x", shoulder=0.7)
        y = tone(np.linspace(0.7, 1.0, 31), look)
        assert np.all(np.diff(y) > 0)                    # still rising: detail kept
        assert np.diff(y)[-1] < np.diff(y)[0]            # but ever more gently

    def test_the_curve_never_turns_back(self):
        for look in (portra400(), vision500t()):
            assert np.all(np.diff(tone(np.linspace(0, 1, 256), look)) >= 0)


class TestGrade:
    def test_skin_keeps_its_hue_while_greens_turn(self):
        look = Look("x", shifts=(Shift(140, 50, turn=-28, chroma=0.7),), chroma=0.9)
        skin, leaf = [0.86, 0.64, 0.52], [0.3, 0.6, 0.25]
        assert abs(hue_chroma(grade(np.array(skin), look))[0] - hue_chroma(skin)[0]) < 3
        assert abs(hue_chroma(grade(np.array(leaf), look))[0] - hue_chroma(leaf)[0]) > 10

    def test_a_band_fades_out_at_its_width(self):
        s = Shift(140, 40)
        assert band(np.array(140.0), s) == 1 and band(np.array(180.0), s) == 0 and band(np.array(100.0), s) == 0

    def test_night_shadows_lean_teal_and_lights_warm(self):
        night = vision500t()
        dark = hue_chroma(grade(np.array([0.12, 0.12, 0.12]), night))[0]
        light = hue_chroma(grade(np.array([0.9, 0.9, 0.9]), night))[0]
        assert 180 < dark < 260              # cyan-blue side
        assert 30 < light < 110              # orange-yellow side

    def test_day_whites_are_warm(self):
        hue, chroma = hue_chroma(grade(np.array([0.92, 0.92, 0.92]), portra400()))
        assert 40 < hue < 110 and chroma > 0.01


class TestCube:
    def test_a_neutral_look_is_the_identity(self):
        x = np.linspace(0, 1, 5)
        table = lut(Look("none"), 5)
        b, g, r = np.meshgrid(x, x, x, indexing="ij")
        assert np.allclose(table, np.stack([r, g, b], -1).reshape(-1, 3), atol=1e-5)

    def test_the_file_says_its_size_and_has_every_entry(self, tmp_path):
        text = write_cube(tmp_path / "x.cube", portra400(), 9).read_text().splitlines()
        assert "LUT_3D_SIZE 9" in text and len([l for l in text if l[0].isdigit()]) == 9 ** 3
