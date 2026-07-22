"""
Figure regression tests.

Every figure the app and the CLI can draw is rendered here from synthetic data and
checked for the structure that makes it the repository's figure rather than a
generic plot: the right number of panels, the physiological floor line, the
per-speed palette, and axis windows that cannot visually exaggerate small
differences. Rendering is done through Matplotlib's Agg backend, so these run
headless and need neither PyMC nor a display.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kinarm_rt import data, filters, figures
from kinarm_rt._speeds import FILTER_WINDOWS, SPEEDS
from kinarm_rt.models import wald, later


@pytest.fixture(scope="module")
def fitted():
    """Filtered trials plus MLE-preview results for both effectors."""
    raw = data.simulate_dataset(n_participants=8, trials_per_cell=80, seed=3)
    tidy = data.load_trials(raw, participant_col="Participant",
                            hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speed_col="Speed_deg_per_s",
                            blocktype_col="BlockType", blocktype_keep="I",
                            rt_units="ms")
    kept, _ = filters.apply_windows(tidy, FILTER_WINDOWS)
    res = {}
    for eff in ("hand", "eye"):
        pv = wald.mle_preview(kept, eff)
        res[eff] = {"group": pv["group"], "units": pv["cell"], "preview": pv}
    return kept, res


def teardown_function(_):
    plt.close("all")


def test_ddm_schematic_one_per_speed(fitted):
    """One conceptual diffusion schematic per speed, each carrying the DDM labels."""
    kept, res = fitted
    figs = figures.ddm_schematic_figs(kept, res["hand"]["group"], "hand")
    assert len(figs) == len(SPEEDS)
    labels, fig = [l for l, _ in figs], figs[0][1]
    assert all("DDM schematic" in l for l in labels)
    text = " ".join(t.get_text() for t in fig.axes[0].texts)
    for token in ("Drift rate", "Response threshold", "Non-decision time", "Baseline"):
        assert token in text, f"schematic is missing the {token!r} annotation"
    plt.close("all")


def test_ddm_schematic_skips_missing_conditions(fitted):
    """A group table with a missing speed yields fewer panels, not a crash."""
    kept, res = fitted
    g = res["hand"]["group"]
    figs = figures.ddm_schematic_figs(kept, g[g["condition"] != 1], "hand")
    assert len(figs) == len(SPEEDS) - 1


def test_ndt_axes_cannot_exaggerate(fitted):
    """
    The saccadic panel must not shrink-wrap the data.

    Saccadic t0 piles on the floor, so a window fitted to the data would span a
    millimetre of t0 and make sub-millisecond noise look like a real effect --
    the same distortion that made the original truncated bars misleading.
    """
    _, res = fitted
    fig = figures.ndt_by_speed(res)
    eye_ax = [a for a in fig.axes if "SRT" in a.get_title()][0]
    lo, hi = eye_ax.get_ylim()
    assert hi - lo >= 60, f"saccadic t0 window is only {hi - lo:.0f} ms wide"
    assert lo <= 70 <= hi, "the 70 ms physiological floor must be in view"
    hand_ax = [a for a in fig.axes if "HRT" in a.get_title()][0]
    hlo, hhi = hand_ax.get_ylim()
    assert hhi - hlo >= 60 and hlo <= 130 <= hhi


def test_ndt_panels_have_floor_line_and_dots(fitted):
    """Each panel draws the floor line plus per-participant dots (not bars)."""
    _, res = fitted
    fig = figures.ndt_by_speed(res)
    for ax in fig.axes:
        assert len(ax.collections) >= len(SPEEDS), "per-participant dots are missing"
        assert not ax.patches, "bars from a truncated baseline should not be used"
        assert any(ln.get_linestyle() in (":", (0, (1, 1.65))) for ln in ax.lines), \
            "the physiological floor line is missing"


def test_why_floors_marks_shape_implied_t0(fitted):
    """The flooring diagnostic shows the implied t0 and the floor for each effector."""
    kept, _ = fitted
    fig = figures.why_floors(kept)
    assert len(fig.axes) == 2
    joined = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "shape-implied" in joined and "physiological" in joined
    assert "skew/CV ratio" in joined


def test_vincentile_suite_has_four_figures(fitted):
    """KDE overlay, histograms, per-speed differences, and the combined view."""
    kept, _ = fitted
    figs = figures.vincentile_suite(kept)
    assert len(figs) == 4
    labels = [l.lower() for l, _ in figs]
    assert any("kde" in l for l in labels)
    assert any("histogram" in l for l in labels)
    assert sum("vincentile" in l for l in labels) == 2


def test_vincentile_difference_is_hand_minus_eye(fitted):
    """Hand RTs are slower, so the difference curve sits above zero."""
    kept, _ = fitted
    diffs = figures._vincentile_shift(kept)
    for c in range(len(SPEEDS)):
        assert len(diffs[c]), f"no vincentile rows for condition {c}"
        assert diffs[c].mean() > 0, "HRT - SRT should be positive"


def test_vincentile_suite_single_effector_fallback(fitted):
    """With only one effector there is nothing to difference, so it degrades gracefully."""
    kept, _ = fitted
    figs = figures.vincentile_suite(kept[kept["effector"] == "hand"])
    assert len(figs) == 1 and "vincentile" in figs[0][0].lower()


def test_reciprobit_panels(fitted):
    """LATER: two reciprobit panels plus latency by speed."""
    kept, _ = fitted
    eye = kept[kept["effector"] == "eye"]
    lat = later.fit_later(eye)
    fig = figures.reciprobit(lat, eye)
    assert len(fig.axes) == 3
    assert "latency" in fig.axes[0].get_xlabel().lower()
    assert "probit" in fig.axes[0].get_ylabel().lower()


def test_reciprobit_handles_empty_later():
    """An empty LATER result returns a placeholder rather than raising."""
    import pandas as pd
    empty = pd.DataFrame()
    fig = figures.reciprobit(
        {"per_participant": empty, "per_cell": empty, "median_r2": float("nan")}, empty)
    assert fig is not None


def test_fit_overlay_one_panel_per_speed(fitted):
    kept, res = fitted
    fig = figures.fit_overlay(kept, "hand", res["hand"]["group"])
    assert len(fig.axes) == len(SPEEDS)


def test_palette_matches_repository():
    """Fills are the repository's per-speed colours; lines are the 0.55 darkened match."""
    from kinarm_rt._speeds import SPEED_RGB
    for c in range(len(SPEEDS)):
        assert figures.FILL[c] == SPEED_RGB[int(SPEEDS[c])]
        assert figures._line(c) == tuple(v * 0.55 for v in figures.FILL[c])


def test_figures_are_vector_safe():
    """PDF/PS text stays editable (fonttype 42), as the repository figures require."""
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
