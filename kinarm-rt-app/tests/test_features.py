"""Tests for the advanced-analysis, stats, export, and pipeline features."""

import numpy as np
import pandas as pd
import pytest

from kinarm_rt import data, filters, analysis, stats_tests, exports
from kinarm_rt.models import wald


@pytest.fixture(scope="module")
def kept():
    d = data.simulate_dataset(n_participants=8, trials_per_cell=100, seed=2)
    tidy = data.load_trials(d, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speed_col="Speed_deg_per_s", blocktype_col="BlockType")
    k, _ = filters.apply_windows(tidy)
    return k


def test_fixed_t0_sensitivity(kept):
    s = analysis.fixed_t0_sensitivity(kept, "eye", t0_values_ms=(50, 70, 90))
    assert set(s["t0_fixed_ms"].unique()) <= {50, 70, 90}
    assert (s["median_ks"] < 0.2).all()          # fit stays good across fixed t0


def test_identifiability_sweep(kept):
    """
    The sweep refits t0 at a range of imposed floors and reports, per cell, the
    slope of fitted t0 against floor (SRT_identifiability_check.py). A slope near 1
    means the floor is setting t0 rather than the data.
    """
    small = kept[kept.participant.isin(sorted(kept.participant.unique())[:2])]
    sw = analysis.identifiability_sweep(small, "eye")
    if sw.empty:
        pytest.skip("no single-component saccade cells in this subset")
    assert {"slope", "tracks_floor"} <= set(sw.columns)
    t0_cols = [c for c in sw.columns if c.startswith("t0_at_")]
    assert len(t0_cols) == 6                      # 40..90 ms, as in the script
    assert sw["tracks_floor"].dtype == bool
    assert ((sw["slope"] > 0.7) == sw["tracks_floor"]).all()
    # a fitted t0 can never sit below the floor that was imposed on it
    for c in t0_cols:
        assert (sw[c] >= int(c.split("_")[-1]) - 1e-6).all()


def test_vincentiles_monotone(kept):
    v = analysis.vincentiles(kept, "hand")
    for c in v["condition"].unique():
        rts = v[v.condition == c].sort_values("quantile")["rt_ms"].values
        assert np.all(np.diff(rts) >= -1e-6)      # quantiles increase


def test_parameter_recovery_signature():
    rec = analysis.parameter_recovery(n_participants=6, trials_per_cell=90)
    assert abs(rec["hand"]["t0_error_ms"]).max() < 25      # hand t0 recovers
    assert rec["eye"]["pct_floored"].mean() > 60           # saccade t0 floors


def test_dissociation(kept):
    hc = wald.mle_preview(kept, "hand")["cell"]
    ec = wald.mle_preview(kept, "eye")["cell"]
    rep = stats_tests.dissociation_report(hc, ec, n_boot=500)
    assert "friedman" in rep["hand"] and "bootstrap" in rep["eye"]


def test_exports_schema(kept):
    res = {"units": pd.DataFrame({"participant": ["P01"], "condition": [0], "speed": [0.0],
                                  "v": [8.0], "a": [0.8], "t0_ms": [160], "t0_lo95": [150],
                                  "t0_hi95": [170], "model": ["single"]}),
           "convergence": {"max_rhat": 1.005, "n_divergences": 0}, "mixture": pd.DataFrame()}
    hrt = exports.to_hrt_fits_csv(res)
    assert list(hrt.columns) == ["pid", "spd", "v", "a", "t0", "t0_lo95", "t0_hi95",
                                 "conv_div", "conv_rhat"]
    srt = exports.to_srt_fits_csv(res)
    assert "express_mode" in srt.columns and "model" in srt.columns


def test_pipeline_preview(tmp_path):
    from kinarm_rt.pipeline import run_pipeline
    out = run_pipeline({"mode": "preview", "out": str(tmp_path / "out"),
                        "analyses": ["vincentiles", "dissociation"]},
                       status=lambda m: None)
    assert (tmp_path / "out" / "report.html").exists()
    assert out["n_trials"] > 0


@pytest.mark.slow
def test_per_speed_hierarchical(kept):
    from kinarm_rt.models import hierarchical
    idata, grp = hierarchical.fit_per_speed(kept, "hand", correlated=False,
                                            draws=150, tune=250, chains=2, cores=1)
    assert len(grp) == 3
    assert {"v", "a", "t0_ms", "t0_ms_lo", "t0_ms_hi"} <= set(grp.columns)
    assert (grp["t0_ms"] > 130).all()                # hand above floor


@pytest.mark.slow
def test_per_speed_lkj(kept):
    from kinarm_rt.models import hierarchical
    idata, grp, corr = hierarchical.fit_per_speed(kept, "hand", correlated=True,
                                                  draws=150, tune=250, chains=2, cores=1)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-6)   # valid correlation matrix
