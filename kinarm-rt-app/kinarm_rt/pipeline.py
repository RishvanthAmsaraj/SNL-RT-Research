"""
Headless pipeline runner: the whole analysis from a config dict, for batch or
cluster use without the GUI. Writes repo-compatible CSVs, an HTML report, a ZIP
bundle, and figures to an output folder.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import data, filters, diagnostics, figures, report, analysis, stats_tests, exports
from .models import wald, later

DEFAULT_CONFIG = {
    "data": None,                       # path to a trial CSV; None -> built-in example
    "mapping": {"participant_col": "Participant", "hand_rt_col": "HandRT_ms",
                "eye_rt_col": "GazeSRT_ms", "speed_col": "Speed_deg_per_s",
                "blocktype_col": "BlockType", "blocktype_keep": "I", "rt_units": "auto"},
    "windows": None,                    # e.g. {"hand": [150,800], "eye": [80,600]} in ms
    "effectors": ["hand", "eye"],
    "mode": "bayesian",                 # "bayesian" | "preview"
    "sampler": {"draws": 1000, "tune": 1000, "chains": 4},
    "use_mixture": True,
    "contamination": 0.0,
    "analyses": ["fixed_t0", "identifiability", "mixture_sensitivity",
                 "vincentiles", "dissociation", "parameter_recovery"],
    "out": "kinarm_out",
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            if path.endswith((".yml", ".yaml")):
                import yaml
                user = yaml.safe_load(f)
            else:
                user = json.load(f)
        cfg.update(user or {})
    return cfg


def run_pipeline(config: dict, status=print) -> dict:
    cfg = dict(DEFAULT_CONFIG); cfg.update(config or {})
    os.makedirs(cfg["out"], exist_ok=True)
    tbl_dir = os.path.join(cfg["out"], "tables"); os.makedirs(tbl_dir, exist_ok=True)
    fig_dir = os.path.join(cfg["out"], "figures"); os.makedirs(fig_dir, exist_ok=True)

    # ---- load ----
    if cfg["data"]:
        status(f"Loading {cfg['data']}")
        raw = data.read_table(cfg["data"])
    else:
        status("No data path; using the built-in example dataset")
        raw = data.simulate_dataset()
    tidy = data.load_trials(raw, **cfg["mapping"])
    windows = None
    if cfg["windows"]:
        windows = {k: (v[0] / 1000, v[1] / 1000) for k, v in cfg["windows"].items()}
    kept, frep = filters.apply_windows(tidy, windows)
    frep.to_csv(os.path.join(tbl_dir, "filter_report.csv"), index=False)
    data.cell_summary(kept).to_csv(os.path.join(tbl_dir, "cell_summary.csv"), index=False)

    results, ctx_figs = {}, {}
    s = cfg["sampler"]

    # ---- fits ----
    for eff in cfg["effectors"]:
        try:
            if cfg["mode"] == "preview":
                prev = wald.mle_preview(kept, eff, cfg["contamination"])
                results[eff] = {"effector": eff, "preview": prev,
                                "group": prev["group"].assign(t0_floor_ms=prev["floor_ms"]),
                                "units": pd.DataFrame(), "mixture": pd.DataFrame(), "convergence": {}}
            else:
                status(f"Bayesian fit: {eff}")
                res = wald.fit_effector(kept, eff, draws=s["draws"], tune=s["tune"],
                                        chains=s["chains"], cores=1,
                                        contamination=cfg["contamination"],
                                        use_mixture=cfg["use_mixture"], status=status)
                res["preview"] = wald.mle_preview(kept, eff, cfg["contamination"])
                res["gof"] = diagnostics.goodness_of_fit(kept, eff, res["units"])
                results[eff] = res
                # repo-compatible CSVs
                if eff == "hand":
                    exports.to_hrt_fits_csv(res).to_csv(os.path.join(tbl_dir, "Bayesian_hrt_fits.csv"), index=False)
                else:
                    exports.to_srt_fits_csv(res).to_csv(os.path.join(tbl_dir, "Bayesian_srt_fits.csv"), index=False)
        except Exception as e:
            status(f"  {eff} fit failed: {e}")

    # ---- LATER ----
    lat = None
    if "eye" in cfg["effectors"]:
        try:
            lat = later.fit_later(kept[kept.effector == "eye"])
            lat["per_cell"].to_csv(os.path.join(tbl_dir, "LATER_fits.csv"), index=False)
            ctx_figs["LATER reciprobit"] = figures.reciprobit(lat, kept[kept.effector == "eye"])
        except Exception as e:
            status(f"  LATER failed: {e}")

    # ---- core figures (same set the app's Graphs tab shows) ----
    ctx_figs["Why saccadic t0 floors"] = figures.why_floors(kept)
    for eff in cfg["effectors"]:
        r = results.get(eff)
        if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
            for lab, f in figures.ddm_schematic_figs(kept, r["group"], eff):
                ctx_figs[f"{eff.capitalize()} — {lab}"] = f
            ctx_figs[f"{eff.capitalize()} fit"] = figures.fit_overlay(kept, eff, r["group"])
    if results:
        ctx_figs["Non-decision time by speed"] = figures.ndt_by_speed(results)

    # ---- analyses ----
    A = cfg["analyses"]
    try:
        if "vincentiles" in A:
            for eff in cfg["effectors"]:
                v = analysis.vincentiles(kept, eff)
                v.to_csv(os.path.join(tbl_dir, f"vincentiles_{eff}.csv"), index=False)
            for lab, f in figures.vincentile_suite(kept):
                ctx_figs[lab] = f
        if "fixed_t0" in A and "eye" in cfg["effectors"]:
            sd = analysis.fixed_t0_sensitivity(kept, "eye")
            sd.to_csv(os.path.join(tbl_dir, "fixed_t0_sensitivity.csv"), index=False)
            ctx_figs["Fixed-t0 sensitivity (eye)"] = figures.fixed_t0_plot(sd, "eye")
        if "identifiability" in A and "eye" in cfg["effectors"]:
            sw = analysis.identifiability_sweep(kept, "eye")
            sw.to_csv(os.path.join(tbl_dir, "identifiability_sweep.csv"), index=False)
            ctx_figs["Identifiability (eye)"] = figures.identifiability_plot(sw, "eye")
        if "mixture_sensitivity" in A:
            analysis.mixture_threshold_sensitivity(kept).to_csv(
                os.path.join(tbl_dir, "mixture_threshold_sensitivity.csv"), index=False)
        if "dissociation" in A:
            uh = results.get("hand", {}).get("units")
            ue = results.get("eye", {}).get("units")
            if uh is None or (isinstance(uh, pd.DataFrame) and uh.empty):
                uh = results.get("hand", {}).get("preview", {}).get("cell")
            if ue is None or (isinstance(ue, pd.DataFrame) and ue.empty):
                ue = results.get("eye", {}).get("preview", {}).get("cell")
            diss = stats_tests.dissociation_report(uh, ue)
            with open(os.path.join(tbl_dir, "dissociation_tests.json"), "w") as f:
                json.dump(diss, f, indent=2, default=str)
            if diss:
                ctx_figs["Non-decision-time dissociation"] = figures.dissociation_plot(diss)
        if "parameter_recovery" in A:
            rec = analysis.parameter_recovery()
            for eff, tb in rec.items():
                tb.to_csv(os.path.join(tbl_dir, f"parameter_recovery_{eff}.csv"), index=False)
    except Exception as e:
        status(f"  an analysis failed: {e}")

    # ---- report + figures ----
    for cap, fig in ctx_figs.items():
        safe = report._safe_name(cap)
        fig.savefig(os.path.join(fig_dir, f"{safe}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(fig_dir, f"{safe}.pdf"), bbox_inches="tight")
    if ctx_figs:
        with open(os.path.join(fig_dir, "all_figures.pdf"), "wb") as f:
            f.write(report.build_figures_pdf(ctx_figs))
    ctx = {"title": "KINARM RT analysis report", "subtitle": "Batch pipeline",
           "filter_report": frep, "cell_summary": data.cell_summary(kept),
           "results": results, "gof": {e: results[e].get("gof") for e in results if results[e].get("gof")},
           "later": lat, "figures": ctx_figs}
    with open(os.path.join(cfg["out"], "report.html"), "w") as f:
        f.write(report.build_html_report(ctx))
    status(f"Done. Outputs in {cfg['out']}/")
    return {"out": cfg["out"], "n_trials": len(kept), "effectors": list(results.keys())}
