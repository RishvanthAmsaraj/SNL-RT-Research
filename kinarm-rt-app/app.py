"""
KINARM reaction-time analysis -- a point-and-click app.

Run it with:   streamlit run app.py

Upload a trial file (or load the example), map the columns, choose a fitting
mode, and download a report with every table and figure. No coding, no IDE.

This reproduces the SNL-RT-Research pipeline: a single-boundary shifted-Wald
fit (drift v, boundary a, non-decision time t0) estimated hierarchically with
partial pooling, an express/regular mixture for bimodal saccade cells, and the
LATER reciprobit model for saccades. The heavy fitting runs in this app's
process; a fast preview mode gives an instant first look.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kinarm_rt import (data, filters, diagnostics, figures, report,
                       analysis, stats_tests, frequentist, compare, exports)
from kinarm_rt.models import wald, later
from kinarm_rt._speeds import EFFECTORS, FILTER_WINDOWS

# PyMC is optional: the app still runs the preview and LATER without it.
try:
    import pymc  # noqa: F401
    HAVE_PYMC = True
except Exception:
    HAVE_PYMC = False
try:
    import diptest  # noqa: F401
    HAVE_DIPTEST = True
except Exception:
    HAVE_DIPTEST = False

st.set_page_config(page_title="KINARM RT analysis", layout="wide", page_icon="🎯")

SS = st.session_state
for k, v in {"raw": None, "tidy": None, "filtered": None, "filter_report": None,
             "results": {}, "later": None, "example": False}.items():
    SS.setdefault(k, v)


def _reset_downstream():
    SS.filtered = None; SS.filter_report = None; SS.results = {}; SS.later = None


# --------------------------------------------------------------------------- #
st.title("KINARM reaction-time analysis")
st.caption("Fit the shifted-Wald and LATER models to hand and saccadic reaction times, "
           "check the fits, and export a report — without opening an IDE.")

if not HAVE_PYMC:
    st.warning("**PyMC is not installed**, so the full Bayesian fit is turned off. The fast "
               "preview and the LATER model still work. To enable Bayesian fitting, install "
               "PyMC (conda recommended): `conda install -c conda-forge pymc arviz`, then "
               "restart the app.")

steps = ["1 · Load data", "2 · Filter", "3 · Fit", "4 · Results & report"]
done = [SS.tidy is not None, SS.filtered is not None, bool(SS.results) or bool(SS.later),
        bool(SS.results) or bool(SS.later)]
for col, label, ok in zip(st.columns(4), steps, done):
    col.markdown(f"**{'✅ ' if ok else ''}{label}**")
st.divider()


# --------------------------------------------------------------------------- Step 1
st.header("1 · Load your trial data")
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload a trial file (CSV/TSV). The repository's wide "
                                "`pooled_data.csv` works directly.", type=["csv", "tsv", "txt"])
with right:
    st.write(""); st.write("")
    if st.button("Load example dataset instead", use_container_width=True):
        SS.raw = data.simulate_dataset(); SS.example = True; _reset_downstream()

if uploaded is not None:
    try:
        SS.raw = data.read_table(uploaded.getvalue()); SS.example = False; 
    except Exception as e:
        st.error(f"Could not read that file: {e}")

raw = SS.raw
if raw is not None:
    st.write(f"Preview — {raw.shape[0]:,} rows, {raw.shape[1]} columns:")
    st.dataframe(raw.head(8), use_container_width=True)

    st.subheader("Map your columns")
    if SS.example:
        st.info("Example data is already in the repository's shape — just press **Use this mapping**.")
    cols = list(raw.columns)

    def pick(label, defaults, allow_none=False):
        opts = (["— none —"] if allow_none else []) + cols
        idx = 0
        for i, cc in enumerate(cols):
            if cc.lower() in defaults:
                idx = i + (1 if allow_none else 0); break
        return st.selectbox(label, opts, index=idx)

    c1, c2, c3 = st.columns(3)
    with c1:
        participant_col = pick("Participant id", {"participant", "subject", "id"})
        layout = st.radio("File layout", ["Wide (one row per trial, both RTs)",
                                          "Long (one RT column + effector column)"])
        rt_units = st.radio("RT units", ["auto-detect", "seconds", "milliseconds"], horizontal=True)
    with c2:
        if layout.startswith("Wide"):
            hand_rt_col = pick("Hand RT column", {"handrt_ms", "handrt", "hand_rt"}, allow_none=True)
            eye_rt_col = pick("Saccade RT column", {"gazesrt_ms", "gazesrt", "saccadert", "eye_rt"}, allow_none=True)
            rt_col = effector_col = effector_value = None
        else:
            rt_col = pick("Reaction-time column", {"rt", "reaction_time", "latency"})
            has_eff = st.checkbox("Has an effector column", value=True)
            if has_eff:
                effector_col = pick("Effector column", {"effector", "modality"}); effector_value = None
            else:
                effector_col = None; effector_value = st.selectbox("These trials are all…", list(EFFECTORS))
            hand_rt_col = eye_rt_col = None
    with c3:
        cond_mode = st.radio("Condition given as",
                             ["SpeedCode (1/2/3)", "raw speed (deg/s)", "condition index (0/1/2)"])
        if cond_mode.startswith("SpeedCode"):
            speedcode_col = pick("SpeedCode column", {"speedcode"}); speed_col = condition_col = None
        elif cond_mode.startswith("raw"):
            speed_col = pick("Speed column", {"speed_deg_per_s", "speed", "target_speed"}); speedcode_col = condition_col = None
        else:
            condition_col = pick("Condition column", {"condition", "cond"}); speed_col = speedcode_col = None
        bt = [c for c in cols if c.lower() in ("blocktype", "block_type", "block")]
        blocktype_col = bt[0] if bt else None
        blocktype_keep = st.text_input("Keep BlockType == (blank = all rows)",
                                       value="I" if blocktype_col else "")

    if st.button("Use this mapping", type="primary"):
        try:
            units = {"auto-detect": "auto", "seconds": "s", "milliseconds": "ms"}[rt_units]
            none = lambda x: None if (x in (None, "— none —")) else x
            tidy = data.load_trials(
                raw, participant_col=participant_col,
                hand_rt_col=none(locals().get("hand_rt_col")),
                eye_rt_col=none(locals().get("eye_rt_col")),
                rt_col=locals().get("rt_col"), effector_col=locals().get("effector_col"),
                effector_value=locals().get("effector_value"),
                condition_col=locals().get("condition_col"), speed_col=locals().get("speed_col"),
                speedcode_col=locals().get("speedcode_col"),
                blocktype_col=blocktype_col, blocktype_keep=(blocktype_keep or None), rt_units=units)
            if tidy.empty:
                st.error("No trials remained after loading. Check the column mapping and BlockType filter.")
            else:
                SS.tidy = tidy; _reset_downstream()
                st.success(f"Loaded {len(tidy):,} trials · {tidy['participant'].nunique()} participants · "
                           f"effectors: {', '.join(sorted(tidy['effector'].unique()))}.")
        except Exception as e:
            st.error(f"Could not load with that mapping: {e}")


# --------------------------------------------------------------------------- Step 2
if SS.tidy is not None:
    st.divider(); st.header("2 · Inclusion windows")
    tidy = SS.tidy
    issues = data.validate(tidy)
    if issues:
        with st.expander(f"Data checks — {len(issues)} note(s)"):
            for it in issues:
                (st.error if it["level"] == "error" else st.warning)(it["message"])
    else:
        st.success("Data checks passed.")

    st.write("Keep trials within these windows (ms). Defaults follow the physiology: "
             "hand 150–800, saccades 80–600.")
    wins = {}
    effs = sorted(tidy["effector"].unique())
    for col, eff in zip(st.columns(max(len(effs), 1)), effs):
        d_lo, d_hi = FILTER_WINDOWS.get(eff, (0.08, 1.5))
        with col:
            lo = st.number_input(f"{eff} lower (ms)", value=int(d_lo * 1000), step=5)
            hi = st.number_input(f"{eff} upper (ms)", value=int(d_hi * 1000), step=25)
            wins[eff] = (lo / 1000, hi / 1000)
    kept, frep = filters.apply_windows(tidy, wins)
    SS.filtered, SS.filter_report = kept, frep
    st.dataframe(frep.round(1), use_container_width=True)
    with st.expander("Distribution shape by condition (why saccadic t₀ floors)"):
        st.write("skew / CV near **3** means near-symmetric for the spread — a shifted Wald cannot "
                 "lift t₀ above the floor. Values well above 3 (typically the hand) support an identified t₀.")
        st.dataframe(data.cell_summary(kept).round(2), use_container_width=True)


# --------------------------------------------------------------------------- Step 3
if SS.filtered is not None:
    st.divider(); st.header("3 · Fit the models")
    kept = SS.filtered
    avail = sorted(kept["effector"].unique())
    c1, c2 = st.columns(2)
    with c1:
        chosen = st.multiselect("Effectors to fit", avail, default=avail)
        modes = ["Quick preview (MLE — seconds)"]
        if HAVE_PYMC:
            modes.append("Full Bayesian (NUTS — minutes)")
        mode = st.radio("Fitting mode", modes)
    with c2:
        preset = st.select_slider("Sampler effort (Bayesian)", ["Fast", "Standard", "Thorough"],
                                  value="Standard", disabled=not HAVE_PYMC)
        draws, tune, chains = {"Fast": (500, 500, 2), "Standard": (1000, 1000, 4),
                               "Thorough": (1500, 1500, 4)}[preset]
        use_mixture = st.checkbox("Fit express/regular mixture for bimodal saccade cells", value=True,
                                  help=("Uses the dip test to flag bimodal cells."
                                        if HAVE_DIPTEST else
                                        "diptest not installed; a Gaussian-mixture BIC fallback is used."))
        contamination = st.slider("Uniform contamination share (0 matches the repo's Bayesian fit)",
                                  0.0, 0.10, 0.0, 0.01)

    if st.button("Run analysis", type="primary"):
        results, errors = {}, []
        if "eye" in chosen:
            try:
                with st.spinner("Fitting LATER to saccades…"):
                    SS.later = later.fit_later(kept[kept.effector == "eye"])
            except Exception as e:
                errors.append(f"LATER: {e}")
        if mode.startswith("Quick"):
            for eff in chosen:
                try:
                    with st.spinner(f"MLE preview — {eff}…"):
                        prev = wald.mle_preview(kept, eff, contamination)
                    results[eff] = {"effector": eff, "preview": prev, "units": pd.DataFrame(),
                                    "group": prev["group"].assign(t0_floor_ms=prev["floor_ms"]),
                                    "mixture": pd.DataFrame(), "convergence": {}}
                except Exception as e:
                    errors.append(f"{eff} preview: {e}")
        else:
            status_box = st.status("Running Bayesian fits…", expanded=True)
            def status(msg): status_box.write(msg)
            for eff in chosen:
                try:
                    res = wald.fit_effector(kept, eff, draws=draws, tune=tune, chains=chains,
                                            cores=1, contamination=contamination,
                                            use_mixture=use_mixture, status=status)
                    res["preview"] = wald.mle_preview(kept, eff, contamination)
                    res["gof"] = diagnostics.goodness_of_fit(kept, eff, res["units"])
                    results[eff] = res
                except Exception as e:
                    errors.append(f"{eff} Bayesian fit: {e}")
            status_box.update(label="Bayesian fits complete", state="complete")
        SS.results = results
        if errors:
            for e in errors:
                st.error(e)
        if results or SS.later:
            st.success("Done. See the results below.")


# --------------------------------------------------------------------------- Step 4
if SS.results or SS.later:
    st.divider(); st.header("4 · Results")
    kept = SS.filtered; res_all = SS.results
    tabs = st.tabs(["Parameters", "Diagnostics", "Figures", "LATER",
                    "Advanced analyses", "Model comparison", "Download"])

    with tabs[0]:
        for eff in EFFECTORS:
            r = res_all.get(eff)
            if not r:
                continue
            st.subheader(eff.capitalize())
            if isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                st.write("Group-level parameters by speed (mean of per-cell estimates):")
                st.dataframe(r["group"].round(2), use_container_width=True)
            if isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                with st.expander(f"{eff} per-participant × speed estimates"):
                    st.dataframe(r["units"], use_container_width=True)
            if isinstance(r.get("mixture"), pd.DataFrame) and len(r["mixture"]):
                st.write("Express/regular mixture cells:")
                st.dataframe(r["mixture"][["participant", "speed", "n", "pi",
                                           "express_mode", "reg_mode"]].round(2),
                             use_container_width=True)

    with tabs[1]:
        shown = False
        for eff in EFFECTORS:
            r = res_all.get(eff)
            if not r or not r.get("convergence"):
                continue
            shown = True
            cv = r["convergence"]; g = r.get("gof", {})
            st.subheader(eff.capitalize())
            m1, m2, m3 = st.columns(3)
            m1.metric("max R-hat", f"{cv.get('max_rhat', float('nan')):.3f}")
            m2.metric("divergences", cv.get("n_divergences", 0))
            m3.metric("median KS", f"{g.get('median_ks', float('nan')):.3f}" if g else "—")
            if not cv.get("converged"):
                st.warning("Not fully converged — raise the sampler effort for a final run.")
            if g and isinstance(g.get("by_condition"), pd.DataFrame) and len(g["by_condition"]):
                st.dataframe(g["by_condition"].round(3), use_container_width=True)
        if not shown:
            st.info("Run the full Bayesian fit to see convergence and goodness-of-fit.")

    with tabs[2]:
        try:
            if SS.later is not None:
                st.pyplot(figures.reciprobit(SS.later, kept[kept.effector == "eye"]))
            st.pyplot(figures.why_floors(kept))
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                    st.pyplot(figures.fit_overlay(kept, eff, r["group"]))
                    src = r["units"] if len(r.get("units", [])) else r.get("preview", {}).get("cell")
                    if src is not None and len(src):
                        floor_ms = r.get("preview", {}).get("floor_ms",
                                    r["group"]["t0_floor_ms"].iloc[0] if "t0_floor_ms" in r["group"] else 130)
                        st.pyplot(figures.ndt_dots(src, r["group"], eff, floor_ms))
        except Exception as e:
            st.error(f"Figure error: {e}")

    with tabs[3]:
        if SS.later is not None:
            lat = SS.later
            st.metric("Median reciprobit R²", f"{lat['median_r2']:.2f}")
            st.write(f"Express-dominant participants: "
                     f"{int(lat['per_participant']['express_dominant'].sum())} of "
                     f"{len(lat['per_participant'])}")
            st.dataframe(lat["per_participant"].round(3), use_container_width=True)
        else:
            st.info("LATER runs when saccadic (eye) trials are included.")

    # -- Advanced analyses (fast; run on demand from the filtered data) --
    with tabs[4]:
        st.caption("These run on the filtered data and are fast (no NUTS needed).")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Non-decision-time dissociation (Friedman + bootstrap + permutation)"):
                uh = (res_all.get("hand", {}) or {}).get("units")
                ue = (res_all.get("eye", {}) or {}).get("units")
                if uh is None or (isinstance(uh, pd.DataFrame) and uh.empty):
                    uh = wald.mle_preview(kept, "hand")["cell"] if "hand" in kept.effector.values else None
                if ue is None or (isinstance(ue, pd.DataFrame) and ue.empty):
                    ue = wald.mle_preview(kept, "eye")["cell"] if "eye" in kept.effector.values else None
                with st.spinner("Running dissociation tests…"):
                    SS["diss"] = stats_tests.dissociation_report(uh, ue)
            if SS.get("diss"):
                for eff, r in SS["diss"].items():
                    st.write(f"**{eff}** — Friedman p = {r['friedman'].get('p_value', float('nan')):.4f}; "
                             f"bootstrap Δt₀ = {r['bootstrap'].get('mean_diff_ms', float('nan')):.1f} ms "
                             f"(95% CI {r['bootstrap'].get('ci95_ms', ['?','?'])[0]:.1f}, "
                             f"{r['bootstrap'].get('ci95_ms', ['?','?'])[1]:.1f}); "
                             f"permutation p = {r['permutation'].get('p_value', float('nan')):.4f}")
                st.pyplot(figures.dissociation_plot(SS["diss"]))
            if st.button("Fixed-t₀ sensitivity (saccades)"):
                with st.spinner("Refitting at fixed t₀ = 50/70/90 ms…"):
                    SS["fixed_t0"] = analysis.fixed_t0_sensitivity(kept, "eye")
            if SS.get("fixed_t0") is not None and len(SS["fixed_t0"]):
                st.dataframe(SS["fixed_t0"].round(3), use_container_width=True)
                st.pyplot(figures.fixed_t0_plot(SS["fixed_t0"], "eye"))
        with a2:
            if st.button("Identifiability sweep (saccades)"):
                with st.spinner("Sweeping the floor…"):
                    SS["ident"] = analysis.identifiability_sweep(kept, "eye")
            if SS.get("ident") is not None and len(SS["ident"]):
                st.pyplot(figures.identifiability_plot(SS["ident"], "eye"))
            if st.button("Mixture-threshold sensitivity"):
                SS["mix_sens"] = analysis.mixture_threshold_sensitivity(kept)
            if SS.get("mix_sens") is not None and len(SS["mix_sens"]):
                st.dataframe(SS["mix_sens"], use_container_width=True)
            if st.button("Vincentiles (model-free)"):
                SS["vinc"] = {e: analysis.vincentiles(kept, e) for e in kept.effector.unique()}
            if SS.get("vinc"):
                for eff, v in SS["vinc"].items():
                    st.pyplot(figures.vincentile_plot(v, eff))
            if st.button("Parameter-recovery study"):
                with st.spinner("Simulating from known parameters and refitting…"):
                    SS["recovery"] = analysis.parameter_recovery()
            if SS.get("recovery"):
                st.caption("Hand t₀ recovers; saccadic t₀ (true ≈ 30 ms) cannot be recovered and pins at the floor.")
                for eff, tb in SS["recovery"].items():
                    st.write(f"**{eff}**"); st.dataframe(tb, use_container_width=True)

    # -- Model comparison (slower; needs PyMC) --
    with tabs[5]:
        if not HAVE_PYMC:
            st.info("Model comparison needs PyMC. Install it and restart to enable this tab.")
        else:
            st.caption("These refit models, so they take a little time.")
            eff_cmp = st.selectbox("Effector", sorted(kept["effector"].unique()), key="cmp_eff")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Compare estimated vs fixed t₀ (LOO)"):
                    box = st.status("Fitting both models for LOO…", expanded=True)
                    try:
                        SS["loo"] = compare.compare_t0_modes(kept, eff_cmp, draws=500, tune=500,
                                                             chains=2, status=box.write)
                        box.update(label="LOO comparison complete", state="complete")
                    except Exception as e:
                        box.update(label="LOO failed", state="error"); st.error(str(e))
                if SS.get("loo"):
                    st.write(f"Preferred model: **{SS['loo']['preferred']}**")
                    st.dataframe(SS["loo"]["table"].round(2), use_container_width=True)
                    st.caption(SS["loo"]["note"])
            with c2:
                if st.button("Frequentist Method A (differential evolution)"):
                    box = st.status(f"Frequentist MLE for {eff_cmp}…", expanded=True)
                    try:
                        SS["freq"] = frequentist.fit_ddm(kept, eff_cmp, status=box.write)
                        box.update(label="Frequentist fit complete", state="complete")
                    except Exception as e:
                        box.update(label="Frequentist fit failed", state="error"); st.error(str(e))
                if SS.get("freq"):
                    st.write(f"Method A (frequentist) group parameters — {eff_cmp}:")
                    st.dataframe(SS["freq"]["group"].round(2), use_container_width=True)
                    r = res_all.get(eff_cmp)
                    if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                        st.caption("Compare with the Bayesian (Method B) group table in the Parameters tab.")

    # -- Download --
    with tabs[6]:
        st.write("Build a self-contained report with every table and figure.")
        ctx = {"title": "KINARM RT analysis report", "subtitle": "Generated by the KINARM RT app",
               "filter_report": SS.filter_report, "cell_summary": data.cell_summary(kept),
               "results": res_all, "gof": {e: res_all[e].get("gof") for e in res_all if res_all[e].get("gof")},
               "later": SS.later, "figures": {}}
        try:
            if SS.later is not None:
                ctx["figures"]["LATER reciprobit"] = figures.reciprobit(SS.later, kept[kept.effector == "eye"])
            ctx["figures"]["Why saccadic t0 floors"] = figures.why_floors(kept)
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                    ctx["figures"][f"{eff.capitalize()} fit"] = figures.fit_overlay(kept, eff, r["group"])
                    src = r["units"] if len(r.get("units", [])) else r.get("preview", {}).get("cell")
                    if src is not None and len(src):
                        floor_ms = r.get("preview", {}).get("floor_ms", 130)
                        ctx["figures"][f"{eff.capitalize()} non-decision time"] = \
                            figures.ndt_dots(src, r["group"], eff, floor_ms)
            html = report.build_html_report(ctx)
            st.download_button("Download HTML report", html, file_name="kinarm_rt_report.html",
                               mime="text/html", use_container_width=True)
            st.download_button("Download full bundle (report + figures + CSVs)",
                               report.build_zip_bundle(ctx), file_name="kinarm_rt_results.zip",
                               mime="application/zip", use_container_width=True)
            # repo-compatible CSVs (drop-in for the pipeline's downstream scripts)
            made_repo_csv = False
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if r and isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                    made_repo_csv = True
                    if eff == "hand":
                        st.download_button("Download Bayesian_hrt_fits.csv (repo format)",
                                           exports.to_hrt_fits_csv(r).to_csv(index=False),
                                           file_name="Bayesian_hrt_fits.csv", mime="text/csv")
                    else:
                        st.download_button("Download Bayesian_srt_fits.csv (repo format)",
                                           exports.to_srt_fits_csv(r).to_csv(index=False),
                                           file_name="Bayesian_srt_fits.csv", mime="text/csv")
            if made_repo_csv:
                st.caption("The repo-format CSVs are drop-in replacements for the pipeline's "
                           "fit tables and feed its downstream figure/NDT scripts.")
        except Exception as e:
            st.error(f"Report error: {e}")
