"""Quality curves as a picture: one panel per clip, the analyzers' scores over time, the picked stretch shaded and,
when given, the stretch a person or an existing edit used outlined. Needs the `plot` extra (matplotlib)."""
from __future__ import annotations

from pathlib import Path

from .scoring import ClipReport

COLORS = {"sharpness": "#2a78d4", "stability": "#d9822b", "exposure": "#7a8a99"}


def plot_reports(reports: list[ClipReport], out: Path, picks: dict[str, tuple[float, float]] | None = None) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("plotting needs matplotlib: uv sync --extra plot") from e

    picks = picks or {}
    fig, axes = plt.subplots(len(reports), 1, figsize=(9, 2.6 * len(reports)), squeeze=False, layout="constrained")
    for ax, rep in zip(axes[:, 0], reports):
        for key, sig in rep.signals.items():
            ax.plot(rep.times, sig, color=COLORS.get(key), lw=1.2, alpha=0.75, label=key)
        ax.plot(rep.times, rep.quality, color="#1b1b1b", lw=2.2, label="quality")
        if rep.best:
            ax.axvspan(*rep.best, color="#3fae5a", alpha=0.18, lw=0,
                       label=f"picked {rep.best[0]:.1f}–{rep.best[1]:.1f}s (q {rep.best_quality:.2f})")
        if rep.name in picks:
            a, b = picks[rep.name]
            ax.axvspan(a, b, fill=False, ec="#c0392b", ls="--", lw=1.5, ymin=0.02, ymax=0.98,
                       label=f"by eye {a:.1f}–{b:.1f}s (q {rep.mean_over(a, b):.2f})")
        ax.set_xlim(0, rep.duration)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(rep.name, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("score")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), fontsize=7.5, ncol=3, frameon=False)
    axes[-1, 0].set_xlabel("seconds")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
