"""
KINARM reaction-time analysis -- a point-and-click app.

Run it with:   streamlit run app.py

Upload a trial file (or load the example), map the columns, choose a fitting
mode, and download a report with every table and figure. No coding, no IDE.

This reproduces the SNL-RT-Research pipeline: a single-boundary shifted-Wald fit
(drift v, boundary a, non-decision time t0) estimated hierarchically with partial
pooling, an express/regular mixture for bimodal saccade cells, and the LATER
reciprobit model for saccades. Heavy fitting runs in this app's process; a fast
preview mode gives an instant first look.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kinarm_rt import (data, filters, diagnostics, figures, report,
                       analysis, stats_tests, frequentist, compare, exports, ui)
from kinarm_rt.models import wald, later, hierarchical
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

st.set_page_config(page_title="KINARM RT analysis", layout="wide", page_icon="🧠")
ui.inject_theme()

SS = st.session_state
for k, v in {"raw": None, "tidy": None, "filtered": None, "filter_report": None,
             "results": {}, "later": None, "example": False}.items():
    SS.setdefault(k, v)


def _reset_downstream():
    SS.filtered = None; SS.filter_report = None; SS.results = {}; SS.later = None
    for k in ("diss", "fixed_t0", "ident", "mix_sens", "vinc", "recovery",
              "loo", "freq", "perspeed"):
        SS.pop(k, None)


# --------------------------------------------------------------------------- header
ui.hero(
    "KINARM Reaction-Time Analysis",
    "Fit the shifted-Wald and LATER models to hand and saccadic reaction times, "
    "check the fits, and export a publication-ready report — no IDE required.",
    badges=["Hierarchical Bayesian · NUTS", "LATER reciprobit",
            "Express / regular mixture", "Reproduces the SNL-RT pipeline"])

_cur = 0 if SS.tidy is None else (2 if not (SS.results or SS.later) else 4)
ui.stepper(["Load data", "Filter", "Fit models", "Results & report"], _cur)

if not HAVE_PYMC:
    st.warning("**PyMC is not installed**, so the full Bayesian fit is off. The fast preview and "
               "LATER still work. Enable Bayesian fitting with "
               "`conda install -c conda-forge pymc arviz`, then restart.")


# --------------------------------------------------------------------------- Step 1
with st.container(border=True):
    ui.section("Load your trial data", "Upload a file or try the example. "
               "The repository's wide pooled_data.csv works directly.", "1")
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Trial file (CSV / TSV), one row per trial",
                                    type=["csv", "tsv", "txt"], label_visibility="collapsed")
    with right:
        if st.button("✨  Load example dataset", use_container_width=True):
            SS.raw = data.simulate_dataset(); SS.example = True; _reset_downstream()

    if uploaded is not None:
        try:
            SS.raw = data.read_table(uploaded.getvalue()); SS.example = False
        except Exception as e:
            st.error(f"Could not read that file: {e}")

    raw = SS.raw
    if raw is not None:
        st.caption(f"Preview — {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(6), use_container_width=True, height=228)

        st.markdown("<div class='kx-pill'>Map your columns</div>", unsafe_allow_html=True)
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

        participant_col = rt_col = effector_col = effector_value = None
        hand_rt_col = eye_rt_col = condition_col = speed_col = speedcode_col = None

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
            else:
                rt_col = pick("Reaction-time column", {"rt", "reaction_time", "latency"})
                if st.checkbox("Has an effector column", value=True):
                    effector_col = pick("Effector column", {"effector", "modality"})
                else:
                    effector_value = st.selectbox("These trials are all…", list(EFFECTORS))
        with c3:
            cond_mode = st.radio("Condition given as",
                                 ["SpeedCode (1/2/3)", "raw speed (deg/s)", "condition index (0/1/2)"])
            if cond_mode.startswith("SpeedCode"):
                speedcode_col = pick("SpeedCode column", {"speedcode"})
            elif cond_mode.startswith("raw"):
                speed_col = pick("Speed column", {"speed_deg_per_s", "speed", "target_speed"})
            else:
                condition_col = pick("Condition column", {"condition", "cond"})
            bt = [c for c in cols if c.lower() in ("blocktype", "block_type", "block")]
            blocktype_col = bt[0] if bt else None
            blocktype_keep = st.text_input("Keep BlockType == (blank = all rows)",
                                           value="I" if blocktype_col else "")

        none = lambda x: None if (x in (None, "— none —")) else x
        if st.button("Use this mapping", type="primary"):
            try:
                u = {"auto-detect": "auto", "seconds": "s", "milliseconds": "ms"}[rt_units]
                tidy = data.load_trials(
                    raw, participant_col=participant_col,
                    hand_rt_col=none(hand_rt_col), eye_rt_col=none(eye_rt_col),
                    rt_col=rt_col, effector_col=effector_col, effector_value=effector_value,
                    condition_col=condition_col, speed_col=speed_col, speedcode_col=speedcode_col,
                    blocktype_col=blocktype_col, blocktype_keep=(blocktype_keep or None), rt_units=u)
                if tidy.empty:
                    st.error("No trials remained. Check the column mapping and BlockType filter.")
                else:
                    SS.tidy = tidy; _reset_downstream()
                    st.success(f"Loaded {len(tidy):,} trials · {tidy['participant'].nunique()} "
                               f"participants · effectors: {', '.join(sorted(tidy['effector'].unique()))}.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load with that mapping: {e}")


# --------------------------------------------------------------------------- Step 2
if SS.tidy is not None:
    with st.container(border=True):
        ui.section("Inclusion windows", "Keep trials within physiological limits "
                   "(hand 150–800 ms, saccades 80–600 ms).", "2")
        tidy = SS.tidy
        issues = data.validate(tidy)
        if issues:
            with st.expander(f"Data checks — {len(issues)} note(s)"):
                for it in issues:
                    (st.error if it["level"] == "error" else st.warning)(it["message"])
        else:
            st.success("Data checks passed.")

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
        with st.expander("Distribution shape by condition — why saccadic t₀ floors"):
            ui.note("skew / CV near <b>3</b> means near-symmetric for the spread — a shifted Wald "
                    "cannot lift t₀ above the floor. Values well above 3 (typically the hand) "
                    "support an identified t₀.")
            st.dataframe(data.cell_summary(kept).round(2), use_container_width=True)


# --------------------------------------------------------------------------- Step 3
if SS.filtered is not None:
    with st.container(border=True):
        ui.section("Fit the models", "A fast preview in seconds, or the full hierarchical "
                   "Bayesian fit in minutes.", "3")
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
            use_mixture = st.checkbox("Express/regular mixture for bimodal saccade cells", value=True,
                                      help=("Uses the dip test to flag bimodal cells." if HAVE_DIPTEST
                                            else "diptest not installed; Gaussian-mixture BIC fallback used."))
            contamination = st.slider("Uniform contamination share (0 matches the repo's Bayesian fit)",
                                      0.0, 0.10, 0.0, 0.01)

        if st.button("▶  Run analysis", type="primary"):
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
                box = st.status("Running Bayesian fits…", expanded=True)
                for eff in chosen:
                    try:
                        res = wald.fit_effector(kept, eff, draws=draws, tune=tune, chains=chains,
                                                cores=1, contamination=contamination,
                                                use_mixture=use_mixture, status=box.write)
                        res["preview"] = wald.mle_preview(kept, eff, contamination)
                        res["gof"] = diagnostics.goodness_of_fit(kept, eff, res["units"])
                        results[eff] = res
                    except Exception as e:
                        errors.append(f"{eff} Bayesian fit: {e}")
                box.update(label="Bayesian fits complete", state="complete")
            SS.results = results
            for e in errors:
                st.error(e)
            if results or SS.later:
                st.success("Done — see the results below.")


# --------------------------------------------------------------------------- Step 4
if SS.results or SS.later:
    with st.container(border=True):
        ui.section("Results & report", "Parameters, diagnostics, figures, advanced analyses, "
                   "and a downloadable report.", "4")
        kept = SS.filtered; res_all = SS.results
        tabs = st.tabs(["📈 Parameters", "🔬 Diagnostics", "🖼 Figures", "👁 LATER",
                        "🧪 Advanced", "⚖️ Model comparison", "⬇️ Download"])

        # -------------------------------------------------- Parameters
        with tabs[0]:
            any_p = False
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if not r:
                    continue
                any_p = True
                st.markdown(f"#### {eff.capitalize()}")
                if isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                    st.caption("Group-level parameters by speed (mean of per-cell estimates)")
                    st.dataframe(r["group"].round(2), use_container_width=True)
                if isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                    with st.expander(f"{eff} per-participant × speed estimates"):
                        st.dataframe(r["units"], use_container_width=True)
                if isinstance(r.get("mixture"), pd.DataFrame) and len(r["mixture"]):
                    st.caption("Express / regular mixture cells")
                    st.dataframe(r["mixture"][["participant", "speed", "n", "pi",
                                               "express_mode", "reg_mode"]].round(2),
                                 use_container_width=True)
            if not any_p:
                st.info("Run a fit to see parameters.")

        # -------------------------------------------------- Diagnostics
        with tabs[1]:
            shown = False
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if not r or not r.get("convergence"):
                    continue
                shown = True
                cv = r["convergence"]; g = r.get("gof", {})
                st.markdown(f"#### {eff.capitalize()}")
                m1, m2, m3 = st.columns(3)
                m1.metric("max R-hat", f"{cv.get('max_rhat', float('nan')):.3f}")
                m2.metric("divergences", cv.get("n_divergences", 0))
                m3.metric("median KS", f"{g.get('median_ks', float('nan')):.3f}" if g else "—")
                if cv.get("converged"):
                    st.success("Converged cleanly (R-hat < 1.01, no divergences).")
                else:
                    st.warning("Not fully converged — raise the sampler effort for a final run.")
                if g and isinstance(g.get("by_condition"), pd.DataFrame) and len(g["by_condition"]):
                    st.dataframe(g["by_condition"].round(3), use_container_width=True)
            if not shown:
                st.info("Run the full Bayesian fit to see convergence and goodness-of-fit.")

        # -------------------------------------------------- Figures
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

        # -------------------------------------------------- LATER
        with tabs[3]:
            if SS.later is not None:
                lat = SS.later
                cA, cB = st.columns(2)
                cA.metric("Median reciprobit R²", f"{lat['median_r2']:.2f}")
                cB.metric("Express-dominant participants",
                          f"{int(lat['per_participant']['express_dominant'].sum())} / "
                          f"{len(lat['per_participant'])}")
                st.dataframe(lat["per_participant"].round(3), use_container_width=True)
            else:
                st.info("LATER runs when saccadic (eye) trials are included.")

        # -------------------------------------------------- Advanced analyses
        with tabs[4]:
            st.caption("Fast analyses on the filtered data — no NUTS needed.")
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Non-decision-time dissociation", use_container_width=True):
                    uh = (res_all.get("hand", {}) or {}).get("units")
                    ue = (res_all.get("eye", {}) or {}).get("units")
                    if uh is None or (isinstance(uh, pd.DataFrame) and uh.empty):
                        uh = wald.mle_preview(kept, "hand")["cell"] if "hand" in kept.effector.values else None
                    if ue is None or (isinstance(ue, pd.DataFrame) and ue.empty):
                        ue = wald.mle_preview(kept, "eye")["cell"] if "eye" in kept.effector.values else None
                    with st.spinner("Friedman + bootstrap + permutation…"):
                        SS["diss"] = stats_tests.dissociation_report(uh, ue)
                if SS.get("diss"):
                    for eff, r in SS["diss"].items():
                        b = r["bootstrap"]
                        st.markdown(f"**{eff}** — Friedman p = {r['friedman'].get('p_value', float('nan')):.4f}; "
                                    f"bootstrap Δt₀ = {b.get('mean_diff_ms', float('nan')):.1f} ms "
                                    f"(95% CI {b.get('ci95_ms', ['?','?'])[0]:.1f}, "
                                    f"{b.get('ci95_ms', ['?','?'])[1]:.1f}); "
                                    f"permutation p = {r['permutation'].get('p_value', float('nan')):.4f}")
                    st.pyplot(figures.dissociation_plot(SS["diss"]))
                if st.button("Fixed-t₀ sensitivity (saccades)", use_container_width=True):
                    with st.spinner("Refitting at fixed t₀ = 50/70/90 ms…"):
                        SS["fixed_t0"] = analysis.fixed_t0_sensitivity(kept, "eye")
                if SS.get("fixed_t0") is not None and len(SS["fixed_t0"]):
                    st.pyplot(figures.fixed_t0_plot(SS["fixed_t0"], "eye"))
                if st.button("Parameter-recovery study", use_container_width=True):
                    with st.spinner("Simulating from known parameters and refitting…"):
                        SS["recovery"] = analysis.parameter_recovery()
                if SS.get("recovery"):
                    ui.note("Hand t₀ recovers; saccadic t₀ (true ≈ 30 ms) cannot be recovered and pins at the floor.")
                    for eff, tb in SS["recovery"].items():
                        st.markdown(f"**{eff}**"); st.dataframe(tb, use_container_width=True)
            with a2:
                if st.button("Identifiability sweep (saccades)", use_container_width=True):
                    with st.spinner("Sweeping the floor…"):
                        SS["ident"] = analysis.identifiability_sweep(kept, "eye")
                if SS.get("ident") is not None and len(SS["ident"]):
                    st.pyplot(figures.identifiability_plot(SS["ident"], "eye"))
                if st.button("Mixture-threshold sensitivity", use_container_width=True):
                    SS["mix_sens"] = analysis.mixture_threshold_sensitivity(kept)
                if SS.get("mix_sens") is not None and len(SS["mix_sens"]):
                    st.dataframe(SS["mix_sens"], use_container_width=True)
                if st.button("Vincentiles (model-free)", use_container_width=True):
                    SS["vinc"] = {e: analysis.vincentiles(kept, e) for e in kept.effector.unique()}
                if SS.get("vinc"):
                    for eff, v in SS["vinc"].items():
                        st.pyplot(figures.vincentile_plot(v, eff))

        # -------------------------------------------------- Model comparison
        with tabs[5]:
            if not HAVE_PYMC:
                st.info("Model comparison needs PyMC. Install it and restart to enable this tab.")
            else:
                st.caption("These refit models, so they take a little time.")
                eff_cmp = st.selectbox("Effector", sorted(kept["effector"].unique()), key="cmp_eff")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Compare estimated vs fixed t₀ (LOO)", use_container_width=True):
                        box = st.status("Fitting both models for LOO…", expanded=True)
                        try:
                            SS["loo"] = compare.compare_t0_modes(kept, eff_cmp, draws=500, tune=500,
                                                                 chains=2, status=box.write)
                            box.update(label="LOO comparison complete", state="complete")
                        except Exception as e:
                            box.update(label="LOO failed", state="error"); st.error(str(e))
                    if SS.get("loo"):
                        st.markdown(f"Preferred model: **{SS['loo']['preferred']}**")
                        st.dataframe(SS["loo"]["table"].round(2), use_container_width=True)
                        st.caption(SS["loo"]["note"])
                with c2:
                    if st.button("Frequentist Method A (diff. evolution)", use_container_width=True):
                        box = st.status(f"Frequentist MLE for {eff_cmp}…", expanded=True)
                        try:
                            SS["freq"] = frequentist.fit_ddm(kept, eff_cmp, status=box.write)
                            box.update(label="Frequentist fit complete", state="complete")
                        except Exception as e:
                            box.update(label="Frequentist fit failed", state="error"); st.error(str(e))
                    if SS.get("freq"):
                        st.caption(f"Method A (frequentist) group parameters — {eff_cmp}")
                        st.dataframe(SS["freq"]["group"].round(2), use_container_width=True)

                st.divider()
                st.markdown("##### Per-speed hierarchical model (group parameters with credible intervals)")
                ui.note("Treats speed as a modelled factor with participant random effects, so you get "
                        "group-level v, a, and t₀ per speed <b>with</b> uncertainty — and optionally "
                        "correlated participant effects (LKJ).")
                correlated = st.checkbox("Model correlated participant effects (LKJ)", value=False)
                if st.button("Fit per-speed hierarchical model", use_container_width=True):
                    box = st.status(f"Fitting per-speed model for {eff_cmp}…", expanded=True)
                    try:
                        box.write("sampling (this treats speed as a factor)…")
                        out = hierarchical.fit_per_speed(kept, eff_cmp, correlated=correlated,
                                                         draws=800, tune=1000, chains=2, cores=1)
                        SS["perspeed"] = {"group": out[1], "corr": out[2] if correlated else None,
                                          "effector": eff_cmp}
                        box.update(label="Per-speed model complete", state="complete")
                    except Exception as e:
                        box.update(label="Per-speed model failed", state="error"); st.error(str(e))
                if SS.get("perspeed"):
                    ps = SS["perspeed"]
                    st.caption(f"Group-level parameters by speed — {ps['effector']} (mean, 94% CI)")
                    st.dataframe(ps["group"].round(2), use_container_width=True)
                    st.pyplot(figures.group_ci_plot(ps["group"], ps["effector"], "t0_ms"))
                    if ps["corr"] is not None:
                        st.caption("Participant-effect correlation matrix (LKJ)")
                        st.dataframe(ps["corr"], use_container_width=True)
                        ui.note("Off-diagonal terms show how participants' parameters covary — "
                                "structure the independent-effects model cannot represent.")

        # -------------------------------------------------- Download
        with tabs[6]:
            st.caption("Build a self-contained report with every table and figure.")
            ctx = {"title": "KINARM RT analysis report", "subtitle": "Generated by the KINARM RT app",
                   "filter_report": SS.filter_report, "cell_summary": data.cell_summary(kept),
                   "results": res_all,
                   "gof": {e: res_all[e].get("gof") for e in res_all if res_all[e].get("gof")},
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
                d1, d2 = st.columns(2)
                d1.download_button("⬇  HTML report", report.build_html_report(ctx),
                                   file_name="kinarm_rt_report.html", mime="text/html",
                                   use_container_width=True)
                d2.download_button("⬇  Full bundle (report + figures + CSVs)",
                                   report.build_zip_bundle(ctx), file_name="kinarm_rt_results.zip",
                                   mime="application/zip", use_container_width=True)
                repo_cols = st.columns(2)
                i = 0
                for eff in EFFECTORS:
                    r = res_all.get(eff)
                    if r and isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                        fn = "Bayesian_hrt_fits.csv" if eff == "hand" else "Bayesian_srt_fits.csv"
                        csv = (exports.to_hrt_fits_csv(r) if eff == "hand"
                               else exports.to_srt_fits_csv(r)).to_csv(index=False)
                        repo_cols[i % 2].download_button(f"⬇  {fn} (repo format)", csv,
                                                         file_name=fn, mime="text/csv",
                                                         use_container_width=True)
                        i += 1
                if i:
                    st.caption("The repo-format CSVs are drop-in replacements for the pipeline's fit "
                               "tables and feed its downstream figure/NDT scripts.")
            except Exception as e:
                st.error(f"Report error: {e}")
