"""
Smoke tests. Run with:  pytest -q
Skip the slow Bayesian test with:  pytest -q -m "not slow"
"""

import numpy as np
import pytest

from kinarm_rt import data, filters, diagnostics
from kinarm_rt.models import wald, later


@pytest.fixture(scope="module")
def kept():
    d = data.simulate_dataset(n_participants=8, trials_per_cell=90, seed=1)
    tidy = data.load_trials(d, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speed_col="Speed_deg_per_s", blocktype_col="BlockType")
    k, _ = filters.apply_windows(tidy)
    return k


def test_wide_load_roundtrip():
    d = data.simulate_dataset(n_participants=4, trials_per_cell=60)
    tidy = data.load_trials(d, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speedcode_col="SpeedCode", blocktype_col="BlockType")
    assert set(tidy["effector"].unique()) == {"hand", "eye"}
    assert (tidy["rt"] > 0).all()
    assert tidy["rt"].median() < 5           # seconds, not milliseconds


def test_filter_windows(kept):
    _, rep = filters.apply_windows(kept)
    assert (rep["pct_kept"] > 90).all()


def test_shape_signature(kept):
    cs = data.cell_summary(kept)
    eye = cs[cs.effector == "eye"]["skew_over_cv"].mean()
    hand = cs[cs.effector == "hand"]["skew_over_cv"].mean()
    assert eye < 6 and hand > eye


def test_mle_preview_flooring(kept):
    h = wald.mle_preview(kept, "hand")
    e = wald.mle_preview(kept, "eye")
    assert h["group"]["t0_ms"].min() > 130          # hand above floor
    assert e["group"]["pct_floored"].mean() > 50     # eye floors


def test_later(kept):
    res = later.fit_later(kept[kept.effector == "eye"])
    assert 0.8 < res["median_r2"] <= 1.0


def test_bimodal_detection_runs(kept):
    eye = kept[kept.effector == "eye"]
    x = eye[(eye.participant == "P01") & (eye.condition == 0)]["rt"].values
    assert isinstance(wald.detect_bimodal(x), bool)


@pytest.mark.slow
def test_bayesian_hand_fit(kept):
    res = wald.fit_effector(kept, "hand", draws=150, tune=250, chains=2, cores=1)
    assert len(res["group"]) == 3
    assert res["convergence"]["n_divergences"] == 0
    t0 = res["group"].sort_values("speed")["t0_ms"].values
    assert (t0 > 130).all()                          # above the hand floor
    gof = diagnostics.goodness_of_fit(kept, "hand", res["units"])
    assert gof["median_ks"] < 0.15


@pytest.mark.slow
def test_bayesian_eye_mixture(kept):
    res = wald.fit_effector(kept, "eye", draws=150, tune=250, chains=2, cores=1, use_mixture=True)
    assert len(res["group"]) == 3
    # saccadic t0 pinned near the 70 ms floor
    assert res["group"]["t0_ms"].max() < 90
