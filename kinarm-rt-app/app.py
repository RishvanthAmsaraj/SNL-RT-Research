"""
KINARM reaction-time analysis -- a point-and-click app.

Run it with:   streamlit run app.py

Upload a trial file (or load the example), map the columns, choose a fitting
mode, and download a report with every table and figure. No coding, no IDE.

Reproduces the SNL-RT-Research pipeline: a single-boundary shifted-Wald fit
(drift v, boundary a, non-decision time t0) estimated hierarchically with partial
pooling, an express/regular mixture for bimodal saccade cells, and the LATER
reciprobit model for saccades.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kinarm_rt import (data, filters, diagnostics, figures, report,
                       analysis, stats_tests, frequentist, compare, exports, ui)
from kinarm_rt.models import wald, later, hierarchical
from kinarm_rt._speeds import EFFECTORS, FILTER_WINDOWS

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


def show_fig(fig):
    """Render a Matplotlib figure as a PNG image and close it.

    Using st.image (a core element) rather than st.pyplot avoids a frontend
    module-loading error some setups hit after long runs, and closing the figure
    keeps memory flat across many reruns. A render failure degrades to a caption
    rather than breaking the page.
    """
    import io
    import matplotlib.pyplot as plt
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, use_container_width=True)
    except Exception as e:
        st.caption(f"(figure could not be displayed: {e})")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


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
    left, right = st.columns([2, 1], vertical_alignment="center")
    with left:
        uploaded = st.file_uploader("Trial file (CSV / TSV), one row per trial",
                                    type=["csv", "tsv", "txt"], label_visibility="collapsed")
    with right:
        if st.button("Load example dataset", use_container_width=True):
            SS.raw = data.simulate_dataset(); SS.example = True; _reset_downstream()

    if uploaded is not None:
        try:
            SS.raw = data.read_table(uploaded.getvalue()); SS.example = False
        except Exception as e:
            st.error(f"Could not read that file: {e}")

    raw = SS.raw
    if raw is not None:
        st.caption(f"Preview — {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(6), use_container_width=True, height=230)

        st.divider()
        ui.eyebrow("Map your columns")
        if SS.example:
            st.info("Example data is already in the repository's shape — just press **Use this mapping**.")
        cols = list(raw.columns)

        def pick(label, defaults, allow_none=False, help=None):
            opts = (["— none —"] if allow_none else []) + cols
            idx = 0
            for i, cc in enumerate(cols):
                if cc.lower() in defaults:
                    idx = i + (1 if allow_none else 0); break
            return st.selectbox(label, opts, index=idx, help=help)

        participant_col = rt_col = effector_col = effector_value = None
        hand_rt_col = eye_rt_col = condition_col = speed_col = speedcode_col = None

        top1, top2 = st.columns(2)
        with top1:
            participant_col = pick("Participant id column", {"participant", "subject", "id"},
                                   help="The column that identifies each participant — one code per "
                                        "person (e.g. CMT001). Every trial is grouped by this, so each "
                                        "person is fit individually and then pooled together.")
        with top2:
            rt_units = st.segmented_control("Reaction-time units",
                                            ["auto-detect", "seconds", "milliseconds"],
                                            default="auto-detect",
                                            help="Whether your reaction times are in seconds (like 0.25) "
                                                 "or milliseconds (like 250). Auto-detect reads the size "
                                                 "of the numbers and picks for you — only override if the "
                                                 "values look off by 1000× after loading.")

        layout = st.segmented_control("How are the reaction times stored?",
                                      ["Wide — both RTs per row", "Long — one RT + effector column"],
                                      default="Wide — both RTs per row",
                                      help="This is about your spreadsheet's shape. WIDE means each trial "
                                           "is one row with a separate column for the hand RT and for the "
                                           "saccade RT (the KINARM export looks like this). LONG means each "
                                           "trial is one row with a single reaction-time column, plus "
                                           "another column that says whether it was a hand or an eye "
                                           "movement. Pick whichever matches your file.")
        m1, m2 = st.columns(2)
        if layout and layout.startswith("Wide"):
            with m1:
                hand_rt_col = pick("Hand RT column", {"handrt_ms", "handrt", "hand_rt"}, allow_none=True,
                                   help="The column with the hand (reach) reaction time — how long after "
                                        "the target appeared the hand began to move. Choose 'none' if your "
                                        "file has no hand data.")
            with m2:
                eye_rt_col = pick("Saccade RT column", {"gazesrt_ms", "gazesrt", "saccadert", "eye_rt"},
                                  allow_none=True,
                                  help="The column with the saccadic (eye) reaction time — how long after "
                                       "the target appeared the eyes began to move. Choose 'none' if your "
                                       "file has no eye data.")
        else:
            with m1:
                rt_col = pick("Reaction-time column", {"rt", "reaction_time", "latency"},
                              help="The single column holding each trial's reaction time — the gap between "
                                   "the target appearing and the response starting.")
            with m2:
                has_eff = st.toggle("There is an effector column", value=True,
                                    help="Turn on if a column labels each trial as a hand or an eye "
                                         "movement. Turn off if every row in the file is the same "
                                         "movement type.")
                if has_eff:
                    effector_col = pick("Effector column", {"effector", "modality"},
                                        help="The column whose values say hand or eye. Common synonyms "
                                             "are accepted (hand/reach, eye/gaze/saccade).")
                else:
                    effector_value = st.segmented_control("These trials are all…", list(EFFECTORS),
                                                          default="hand")

        cond_mode = st.segmented_control("How is target speed given?",
                                         ["SpeedCode (1/2/3)", "Speed (deg/s)", "Condition index (0/1/2)"],
                                         default="SpeedCode (1/2/3)",
                                         help="Each trial belongs to a target-speed condition; this study "
                                              "uses three — 0, 75, and 150 degrees per second. Tell the app "
                                              "how that's encoded in your file: a SpeedCode of 1/2/3 (which "
                                              "maps to 0/75/150), the actual speed in deg/s, or a 0/1/2 "
                                              "condition index. They all end up as the same three conditions.")
        s1, s2 = st.columns(2)
        with s1:
            if cond_mode and cond_mode.startswith("SpeedCode"):
                speedcode_col = pick("SpeedCode column", {"speedcode"})
            elif cond_mode and cond_mode.startswith("Speed"):
                speed_col = pick("Speed column", {"speed_deg_per_s", "speed", "target_speed"})
            else:
                condition_col = pick("Condition column", {"condition", "cond"})
        with s2:
            bt = [c for c in cols if c.lower() in ("blocktype", "block_type", "block")]
            blocktype_col = bt[0] if bt else None
            blocktype_keep = st.text_input(
                "Keep only this block type (blank = keep all)",
                value="I" if blocktype_col else "",
                help="Experiments often mix different task blocks in one file. Here the BlockType column "
                     "marks each trial's block — 'I' is the interception task you want to analyse, and "
                     "other letters (like 'S') are different blocks that should be left out. Type the "
                     "letter to keep (usually I), or leave this blank to keep every trial.")

        none = lambda x: None if (x in (None, "— none —")) else x
        if st.button("Use this mapping", type="primary"):
            try:
                u = {"auto-detect": "auto", "seconds": "s", "milliseconds": "ms"}[rt_units or "auto-detect"]
                tidy = data.load_trials(
                    raw, participant_col=participant_col,
                    hand_rt_col=none(hand_rt_col), eye_rt_col=none(eye_rt_col),
                    rt_col=rt_col, effector_col=effector_col, effector_value=effector_value,
                    condition_col=condition_col, speed_col=speed_col, speedcode_col=speedcode_col,
                    blocktype_col=blocktype_col, blocktype_keep=(blocktype_keep or None), rt_units=u)
                if tidy.empty:
                    st.error("No trials remained. Check the column mapping and block-type filter.")
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

        ui.eyebrow("Windows (milliseconds)")
        wins = {}
        effs = sorted(tidy["effector"].unique())
        for col, eff in zip(st.columns(max(len(effs), 1)), effs):
            d_lo, d_hi = FILTER_WINDOWS.get(eff, (0.08, 1.5))
            with col:
                cc = st.columns(2)
                lo = cc[0].number_input(f"{eff} — lower", value=int(d_lo * 1000), step=5)
                hi = cc[1].number_input(f"{eff} — upper", value=int(d_hi * 1000), step=25)
                wins[eff] = (lo / 1000, hi / 1000)
        kept, frep = filters.apply_windows(tidy, wins)
        SS.filtered, SS.filter_report = kept, frep

        ui.eyebrow("Kept after filtering")
        st.dataframe(frep.round(1), use_container_width=True, hide_index=True)
        with st.expander("Distribution shape by condition — why saccadic t₀ floors"):
            ui.note("skew / CV near <b>3</b> means near-symmetric for the spread — a shifted Wald "
                    "cannot lift t₀ above the floor. Values well above 3 (typically the hand) "
                    "support an identified t₀.")
            st.dataframe(data.cell_summary(kept).round(2), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- Step 3
if SS.filtered is not None:
    with st.container(border=True):
        ui.section("Fit the models", "A fast preview in seconds, or the full hierarchical "
                   "Bayesian fit in minutes.", "3")
        kept = SS.filtered
        avail = sorted(kept["effector"].unique())
        c1, c2 = st.columns(2)
        with c1:
            ui.eyebrow("What to fit")
            chosen = st.segmented_control("Effectors", avail, selection_mode="multi",
                                          default=avail, key="fit_eff",
                                          help="Which movement types to model — hand, eye, or both. "
                                               "Each is fit and reported separately, because hand and eye "
                                               "reaction times behave quite differently.")
            modes = ["Quick preview (seconds)"]
            if HAVE_PYMC:
                modes.append("Full Bayesian (minutes)")
            mode = st.segmented_control("Fitting mode", modes, default=modes[0],
                                        help="Preview runs a quick best-fit estimate in seconds — great "
                                             "for a first look and for checking the data loaded correctly. "
                                             "Full Bayesian runs the proper hierarchical model with a "
                                             "sampler (NUTS): it takes a few minutes but gives uncertainty "
                                             "(credible intervals) and is the version you'd report.")
        with c2:
            ui.eyebrow("Bayesian options")
            preset = st.select_slider("Sampler effort", ["Fast", "Standard", "Thorough"],
                                      value="Standard", disabled=not HAVE_PYMC,
                                      help="How hard the sampler works. More draws and chains give more "
                                           "accurate, stable estimates but take longer. Fast (500/500/2) "
                                           "for trying things out, Standard (1000/1000/4) for most runs, "
                                           "Thorough (1500/1500/4) to match the paper for a final result. "
                                           "The numbers are draws / tuning steps / parallel chains.")
            draws, tune, chains = {"Fast": (500, 500, 2), "Standard": (1000, 1000, 4),
                                   "Thorough": (1500, 1500, 4)}[preset]
            use_mixture = st.toggle("Express/regular mixture for bimodal saccade cells", value=True,
                                    help=("Some people make very fast 'express' saccades mixed in with "
                                          "normal ones, so their reaction-time distribution has two humps. "
                                          "When on, the app detects those cells (with a statistical dip "
                                          "test) and fits two pieces — a fast express one and a regular "
                                          "one — instead of forcing a single curve through both."
                                          if HAVE_DIPTEST else
                                          "Fits two components (fast express + regular) for saccade cells "
                                          "with two humps. diptest isn't installed, so a Gaussian-mixture "
                                          "BIC test is used to flag them instead."))
            contamination = st.slider("Uniform contamination share", 0.0, 0.10, 0.0, 0.01,
                                      help="A few trials are always flukes (lapses, tracker glitches) that "
                                           "no model fits well. This adds a flat 'anything-goes' component "
                                           "to soak them up so they don't distort the estimates. 0 turns it "
                                           "off, which matches the repository's Bayesian setup; a few "
                                           "percent makes the fit more robust to outliers.")

        chosen = chosen or []
        if st.button("Run analysis", type="primary", disabled=not chosen):
            results, errors = {}, []
            if "eye" in chosen:
                try:
                    with st.spinner("Fitting LATER to saccades…"):
                        SS.later = later.fit_later(kept[kept.effector == "eye"])
                except Exception as e:
                    errors.append(f"LATER: {e}")
            if mode and mode.startswith("Quick"):
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


# --------------------------------------------------------------------------- fragments
@st.fragment
def advanced_tab():
    kept = SS.filtered; res_all = SS.results
    ui.hint("Fast analyses on the filtered data — no sampling needed. Results appear inline.")
    a1, a2 = st.columns(2)
    with a1:
        ui.eyebrow("Statistics")
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
            show_fig(figures.dissociation_plot(SS["diss"]))
        if st.button("Parameter-recovery study", use_container_width=True):
            with st.spinner("Simulating from known parameters and refitting…"):
                SS["recovery"] = analysis.parameter_recovery()
        if SS.get("recovery"):
            ui.note("Hand t₀ recovers; saccadic t₀ (true ≈ 30 ms) cannot be recovered and pins at the floor.")
            for eff, tb in SS["recovery"].items():
                st.markdown(f"**{eff}**"); st.dataframe(tb, use_container_width=True, hide_index=True)
        if st.button("Mixture-threshold sensitivity", use_container_width=True):
            SS["mix_sens"] = analysis.mixture_threshold_sensitivity(kept)
        if SS.get("mix_sens") is not None and len(SS["mix_sens"]):
            st.dataframe(SS["mix_sens"], use_container_width=True, hide_index=True)
    with a2:
        ui.eyebrow("Graphs")
        if st.button("Fixed-t₀ sensitivity (saccades)", use_container_width=True):
            with st.spinner("Refitting at fixed t₀ = 50/70/90 ms…"):
                SS["fixed_t0"] = analysis.fixed_t0_sensitivity(kept, "eye")
        if SS.get("fixed_t0") is not None and len(SS["fixed_t0"]):
            show_fig(figures.fixed_t0_plot(SS["fixed_t0"], "eye"))
        if st.button("Identifiability sweep (saccades)", use_container_width=True):
            with st.spinner("Sweeping the floor…"):
                SS["ident"] = analysis.identifiability_sweep(kept, "eye")
        if SS.get("ident") is not None and len(SS["ident"]):
            show_fig(figures.identifiability_plot(SS["ident"], "eye"))
        if st.button("Vincentiles (model-free)", use_container_width=True):
            SS["vinc"] = {e: analysis.vincentiles(kept, e) for e in kept.effector.unique()}
        if SS.get("vinc"):
            for eff, v in SS["vinc"].items():
                show_fig(figures.vincentile_plot(v, eff))


@st.fragment
def comparison_tab():
    kept = SS.filtered
    if not HAVE_PYMC:
        st.info("Model comparison needs PyMC. Install it and restart to enable this tab.")
        return
    ui.hint("These refit models, so they take a little time. Results appear inline.")
    eff_cmp = st.segmented_control("Effector", sorted(kept["effector"].unique()),
                                   default=sorted(kept["effector"].unique())[0], key="cmp_eff")
    eff_cmp = eff_cmp or sorted(kept["effector"].unique())[0]
    c1, c2 = st.columns(2)
    with c1:
        ui.eyebrow("Cross-validation")
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
            st.dataframe(SS["loo"]["table"].round(2), use_container_width=True, hide_index=True)
            st.caption(SS["loo"]["note"])
    with c2:
        ui.eyebrow("Frequentist check")
        if st.button("Method A (differential evolution)", use_container_width=True):
            box = st.status(f"Frequentist MLE for {eff_cmp}…", expanded=True)
            try:
                SS["freq"] = frequentist.fit_ddm(kept, eff_cmp, status=box.write)
                box.update(label="Frequentist fit complete", state="complete")
            except Exception as e:
                box.update(label="Frequentist fit failed", state="error"); st.error(str(e))
        if SS.get("freq"):
            st.caption(f"Method A group parameters — {eff_cmp}")
            st.dataframe(SS["freq"]["group"].round(2), use_container_width=True, hide_index=True)

    st.divider()
    ui.eyebrow("Per-speed hierarchical model (group parameters with credible intervals)")
    ui.note("Treats speed as a modelled factor with participant random effects, so you get "
            "group-level v, a, and t₀ per speed <b>with</b> uncertainty — and optionally "
            "correlated participant effects (LKJ).")
    correlated = st.toggle("Model correlated participant effects (LKJ)", value=False)
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
        st.dataframe(ps["group"].round(2), use_container_width=True, hide_index=True)
        show_fig(figures.group_ci_plot(ps["group"], ps["effector"], "t0_ms"))
        if ps["corr"] is not None:
            st.caption("Participant-effect correlation matrix (LKJ)")
            st.dataframe(ps["corr"], use_container_width=True)
            ui.note("Off-diagonal terms show how participants' parameters covary — "
                    "structure the independent-effects model cannot represent.")


# --------------------------------------------------------------------------- Step 4
if SS.results or SS.later:
    with st.container(border=True):
        ui.section("Results & report", "Parameters, diagnostics, graphs, advanced analyses, "
                   "and a downloadable report.", "4")
        kept = SS.filtered; res_all = SS.results
        tabs = st.tabs(["Parameters", "Diagnostics", "Graphs", "LATER",
                        "Advanced", "Model comparison", "Download"])

        # -------------------------------------------------- Parameters
        with tabs[0]:
            ui.hint("Fitted model parameters as tables. Drift v, boundary a, and non-decision "
                    "time t₀ (ms) by target speed.")
            any_p = False
            for eff in EFFECTORS:
                r = res_all.get(eff)
                if not r:
                    continue
                any_p = True
                st.markdown(f"#### {eff.capitalize()}")
                if isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                    ui.eyebrow("Group parameters by speed")
                    st.dataframe(r["group"].round(2), use_container_width=True, hide_index=True)
                if isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                    with st.expander(f"{eff} — per-participant × speed estimates"):
                        st.dataframe(r["units"], use_container_width=True, hide_index=True)
                if isinstance(r.get("mixture"), pd.DataFrame) and len(r["mixture"]):
                    ui.eyebrow("Express / regular mixture cells")
                    st.dataframe(r["mixture"][["participant", "speed", "n", "pi",
                                               "express_mode", "reg_mode"]].round(2),
                                 use_container_width=True, hide_index=True)
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
                    st.dataframe(g["by_condition"].round(3), use_container_width=True, hide_index=True)
            if not shown:
                st.info("Run the full Bayesian fit to see convergence and goodness-of-fit.")

        # -------------------------------------------------- Graphs
        with tabs[2]:
            ui.hint("Visual model checks and summaries.")
            try:
                if SS.later is not None:
                    show_fig(figures.reciprobit(SS.later, kept[kept.effector == "eye"]))
                show_fig(figures.why_floors(kept))
                for eff in EFFECTORS:
                    r = res_all.get(eff)
                    if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                        show_fig(figures.fit_overlay(kept, eff, r["group"]))
                        src = r["units"] if len(r.get("units", [])) else r.get("preview", {}).get("cell")
                        if src is not None and len(src):
                            floor_ms = r.get("preview", {}).get("floor_ms",
                                        r["group"]["t0_floor_ms"].iloc[0] if "t0_floor_ms" in r["group"] else 130)
                            show_fig(figures.ndt_dots(src, r["group"], eff, floor_ms))
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
                st.dataframe(lat["per_participant"].round(3), use_container_width=True, hide_index=True)
            else:
                st.info("LATER runs when saccadic (eye) trials are included.")

        # -------------------------------------------------- Advanced / Comparison (fragments)
        with tabs[4]:
            advanced_tab()
        with tabs[5]:
            comparison_tab()

        # -------------------------------------------------- Download
        with tabs[6]:
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

                ui.eyebrow("Report & figures")
                ui.hint("HTML report to read in a browser · a vector PDF of every figure · or a full "
                        "ZIP with the report, figures (PNG + PDF), and all tables as CSV.")
                d1, d2, d3 = st.columns(3)
                d1.download_button("HTML report", report.build_html_report(ctx),
                                   file_name="kinarm_rt_report.html", mime="text/html",
                                   use_container_width=True)
                d2.download_button("Figures (PDF)", report.build_figures_pdf(ctx["figures"]),
                                   file_name="kinarm_rt_figures.pdf", mime="application/pdf",
                                   use_container_width=True)
                d3.download_button("Full bundle (ZIP)", report.build_zip_bundle(ctx),
                                   file_name="kinarm_rt_results.zip", mime="application/zip",
                                   use_container_width=True)

                have_units = [e for e in EFFECTORS if res_all.get(e)
                              and isinstance(res_all[e].get("units"), pd.DataFrame)
                              and len(res_all[e]["units"])]
                if have_units:
                    ui.eyebrow("Repository-format tables")
                    ui.hint("Drop-in replacements for the pipeline's fit tables; they feed its "
                            "downstream figure and NDT scripts unchanged.")
                    rc = st.columns(max(len(have_units), 1))
                    for col, eff in zip(rc, have_units):
                        fn = "Bayesian_hrt_fits.csv" if eff == "hand" else "Bayesian_srt_fits.csv"
                        csv = (exports.to_hrt_fits_csv(res_all[eff]) if eff == "hand"
                               else exports.to_srt_fits_csv(res_all[eff])).to_csv(index=False)
                        col.download_button(fn, csv, file_name=fn, mime="text/csv",
                                            use_container_width=True)
            except Exception as e:
                st.error(f"Report error: {e}")
