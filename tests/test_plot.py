from pathlib import Path

import pytest

from aicut_analyzer.analyzers import ExposureAnalyzer, SharpnessAnalyzer, StabilityAnalyzer
from aicut_analyzer.scoring import QualityScorer, WindowPicker
from synthetic import as_clip, footage

pytest.importorskip("matplotlib")


def test_draws_a_panel_per_clip(tmp_path: Path):
    from aicut_analyzer.plot import plot_reports

    scorer = QualityScorer([(SharpnessAnalyzer(), 0.45), (StabilityAnalyzer(), 0.40), (ExposureAnalyzer(), 0.15)])
    reports = [WindowPicker().pick(scorer.report(as_clip(footage(45, blur=slice(10, 20)), name=f"c{i}.mp4")))
               for i in range(2)]
    out = plot_reports(reports, tmp_path / "curves.png", picks={"c0.mp4": (0.5, 2.5)})
    assert out.exists() and out.stat().st_size > 10_000
