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

import io

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


def fig_to_png(fig) -> bytes:
    """
    Render a Matplotlib figure to PNG bytes and release it.

    PNG through st.image rather than st.pyplot avoids a frontend module-loading
    error some setups hit after long runs, and closing the figure keeps memory flat
    across many reruns. 110 dpi stays crisp at typical column widths -- still about
    twice the displayed pixel size -- without shipping megabytes to the browser.
    """
    import matplotlib.pyplot as plt
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        return buf.getvalue()
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def show_fig(fig):
    """Draw a figure straight away (used where caching would not help)."""
    try:
        st.image(fig_to_png(fig), width="stretch")
    except Exception as e:
        st.caption(f"(figure could not be displayed: {e})")


def results_signature() -> str:
    """
    A fingerprint of everything the figures are drawn from.

    Streamlit re-runs the whole script on every interaction and renders every tab
    body, selected or not, so without this the figures are rebuilt each time a
    slider moves -- several seconds of work for a result that has not changed.
    """
    import hashlib
    parts = [str(len(SS.get("kept", []))), str(SS.get("win_hand")), str(SS.get("win_eye"))]
    for eff in EFFECTORS:
        r = (SS.results or {}).get(eff)
        if not r:
            continue
        for k in ("group", "units", "mixture"):
            v = r.get(k)
            if isinstance(v, pd.DataFrame) and len(v):
                parts.append(f"{eff}:{k}:{pd.util.hash_pandas_object(v, index=True).sum()}")
        pv = r.get("preview") or {}
        cell = pv.get("cell")
        if isinstance(cell, pd.DataFrame) and len(cell):
            parts.append(f"{eff}:prev:{pd.util.hash_pandas_object(cell, index=True).sum()}")
    if SS.later:
        parts.append(f"later:{SS.later.get('median_r2')}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def cached_figs(name: str, build):
    """
    Render a group of figures once for a given set of results, then reuse them.

    `build` returns a list of (label, figure). The rendered PNGs are kept until the
    results change, at which point the whole cache is dropped so nothing stale can
    survive a refit.
    """
    sig = results_signature()
    store = SS.setdefault("_figcache", {})
    if store.get("_sig") != sig:
        store.clear()
        store["_sig"] = sig
    if name not in store:
        store[name] = [(lab, fig_to_png(f)) for lab, f in build()]
    return store[name]


def df_key(df) -> str:
    """A short fingerprint of a table, for keying a cached figure to its data."""
    try:
        return str(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        return str(len(df))


def show_cached(name: str, build):
    """Show a cached group of figures, drawing them only the first time."""
    try:
        for _lab, png in cached_figs(name, build):
            st.image(png, width="stretch")
    except Exception as e:
        st.caption(f"(figures could not be displayed: {e})")


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
        if st.button("Load example dataset", width="stretch"):
            SS.raw = data.simulate_dataset(); SS.example = True; _reset_downstream()

    if uploaded is not None:
        try:
            SS.raw = data.read_table(uploaded.getvalue()); SS.example = False
        except Exception as e:
            st.error(f"Could not read that file: {e}")

    raw = SS.raw
    if raw is not None:
        st.caption(f"Preview — {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        st.dataframe(raw.head(6), width="stretch", height=230)

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
        st.dataframe(frep.round(1), width="stretch", hide_index=True)
        with st.expander("Distribution shape by condition — why saccadic t₀ floors"):
            ui.hint("skew / CV near <b>3</b> means near-symmetric for the spread — a shifted Wald "
                    "cannot lift t₀ above the floor. Values well above 3 (typically the hand) "
                    "support an identified t₀.")
            st.dataframe(data.cell_summary(kept).round(2), width="stretch", hide_index=True)


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
            modes = ["Method A — frequentist MLE"]
            if HAVE_PYMC:
                modes.append("Method B — hierarchical Bayesian")
            mode = st.segmented_control("Fitting mode", modes, default=modes[0],
                                        help="The two methods from the pipeline. Method A fits each "
                                             "participant × speed cell on its own by maximum likelihood "
                                             "(this is DDM_fit.py, same optimiser and settings) and "
                                             "takes a minute or two. "
                                             "Method B is the hierarchical model sampled with NUTS "
                                             "(Bayesian_HRT_fit.py / Bayesian_SRT_fit.py): it borrows "
                                             "strength across participants and gives credible intervals, "
                                             "and is the version reported in the write-up. Running both "
                                             "and showing they agree is the standard robustness check.")
        with c2:
            ui.eyebrow("Bayesian options")
            preset = st.select_slider("Sampler effort", ["Fast", "Standard", "Thorough"],
                                      value="Thorough", disabled=not HAVE_PYMC,
                                      help="How hard the sampler works, as draws / tuning steps / chains. "
                                           "Thorough (1500/1500/4) is what the pipeline scripts use, so it "
                                           "is the default here and the setting to report. Standard "
                                           "(1000/1000/4) and Fast (500/500/2) are for trying things out — "
                                           "they reach the same estimates but with more sampling noise.")
            draws, tune, chains = {"Fast": (500, 500, 2), "Standard": (1000, 1000, 4),
                                   "Thorough": (1500, 1500, 4)}[preset]
            use_mixture = st.toggle("Two-component fit for saccade cells that need it", value=True,
                                    help=("Some saccade cells have two humps rather than one, and a single "
                                          "shifted-Wald cannot fit both. Where that happens the pipeline fits "
                                          "two components — a faster and a slower one — keeping the "
                                          "two-component version only if the single fit was poor (KS > 0.10), "
                                          "the two-component fit is good (KS < 0.10), it splits the trials "
                                          "between 10% and 90%, and the two modes are at least 30 ms apart. "
                                          "This is a statistical split, not a claim about express saccades: "
                                          "in the reference dataset only one of the sixteen two-component "
                                          "cells has its fast mode under 130 ms. Use the LATER tab for the "
                                          "express question. Off means every cell gets a single component."))
            contamination = st.slider("Uniform contamination share", 0.0, 0.10, 0.0, 0.01,
                                      help="A few trials are always flukes (lapses, tracker glitches) that "
                                           "no model fits well. This adds a flat 'anything-goes' component "
                                           "to soak them up so they don't distort the estimates. 0 turns it "
                                           "off, which matches the repository's Bayesian setup; a few "
                                           "percent makes the fit more robust to outliers.")


        chosen = chosen or []
        if st.button("Run analysis", type="primary", disabled=not chosen):
            results, errors = {}, []
            is_method_a = bool(mode and mode.startswith("Method A"))
            n_jobs = wald.default_jobs()

            # Lay out every stage before any of them starts, so the panel shows what
            # the run consists of and which part is moving.
            stages = []
            if "eye" in chosen:
                stages.append(("later", "LATER model — saccades"))
            for eff in chosen:
                name = "hand" if eff == "hand" else "saccade"
                if is_method_a:
                    stages.append((f"a_{eff}", f"Method A — {name}"))
                else:
                    stages.append((f"bayes_{eff}", f"Method B sampling — {name}"))
                    stages.append((f"a_{eff}", f"Method A cross-check — {name}"))

            box = st.status(f"Running {len(stages)} stages…", expanded=True)
            box.caption(f"Fitting cells on {n_jobs} worker thread{'s' if n_jobs > 1 else ''}. "
                        f"Cells are independent and seeded individually, so this affects only "
                        f"how long the run takes, never the result.")
            prog = ui.RunProgress(box, stages)

            if "eye" in chosen:
                try:
                    prog["later"].note("fitting reciprobit lines")
                    SS.later = later.fit_later(kept[kept.effector == "eye"])
                    npp = len(SS.later.get("per_participant", []))
                    prog.finish_stage("later", f"{npp} participants" if npp else "")
                except Exception as e:
                    prog.fail_stage("later", str(e)[:60])
                    errors.append(f"LATER: {e}")

            for eff in chosen:
                if not is_method_a:
                    try:
                        bar = prog[f"bayes_{eff}"]
                        res = wald.fit_effector(kept, eff, draws=draws, tune=tune, chains=chains,
                                                cores=1, contamination=contamination,
                                                use_mixture=use_mixture, status=bar.note,
                                                n_jobs=n_jobs, progress=bar)
                        prog.finish_stage(f"bayes_{eff}")
                        results[eff] = res
                    except Exception as e:
                        prog.fail_stage(f"bayes_{eff}", str(e)[:60])
                        errors.append(f"{eff} Bayesian fit: {e}")
                try:
                    bar = prog[f"a_{eff}"]
                    prev = wald.mle_preview(kept, eff, contamination, use_mixture=use_mixture,
                                            n_jobs=n_jobs, progress=bar, status=bar.note,
                                            selection=(results.get(eff) or {}).get("selection"))
                    nmix = int((prev["cell"]["model"] == "mixture").sum()) if len(prev["cell"]) else 0
                    prog.finish_stage(f"a_{eff}",
                                      f"{nmix} cells needed two components" if nmix else "")
                    if is_method_a:
                        results[eff] = {"effector": eff, "preview": prev, "units": pd.DataFrame(),
                                        "group": prev["group"].assign(t0_floor_ms=prev["floor_ms"]),
                                        "mixture": pd.DataFrame(), "convergence": {}}
                    elif eff in results:
                        results[eff]["preview"] = prev
                        results[eff]["gof"] = diagnostics.goodness_of_fit(kept, eff,
                                                                         results[eff]["units"])
                except Exception as e:
                    prog.fail_stage(f"a_{eff}", str(e)[:60])
                    errors.append(f"{eff} Method A: {e}")

            prog.finish()
            box.update(label="Analysis complete" if not errors else "Finished with errors",
                       state="complete" if not errors else "error", expanded=False)
            # anything prepared for download describes the previous fit
            for _k in ("html", "pdf", "zip"):
                SS.pop(f"dl_{_k}", None)
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
        if st.button("Non-decision-time dissociation", width="stretch"):
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
            show_cached("diss:" + str(sorted(SS["diss"])),
                        lambda: [("dissociation", figures.dissociation_plot(SS["diss"]))])
        if st.button("Parameter-recovery study", width="stretch"):
            with st.spinner("Simulating from known parameters and refitting…"):
                SS["recovery"] = analysis.parameter_recovery()
        if SS.get("recovery"):
            ui.hint("Hand t₀ recovers; saccadic t₀ (true ≈ 30 ms) cannot be recovered and pins at the floor.")
            for eff, tb in SS["recovery"].items():
                st.markdown(f"**{eff}**"); st.dataframe(tb, width="stretch", hide_index=True)
        if st.button("Mixture-threshold sensitivity", width="stretch"):
            SS["mix_sens"] = analysis.mixture_threshold_sensitivity(kept)
        if SS.get("mix_sens") is not None and len(SS["mix_sens"]):
            st.dataframe(SS["mix_sens"], width="stretch", hide_index=True)
    with a2:
        ui.eyebrow("Graphs")
        if st.button("Fixed-t₀ sensitivity (saccades)", width="stretch"):
            with st.spinner("Refitting at fixed t₀ = 50/70/90 ms…"):
                SS["fixed_t0"] = analysis.fixed_t0_sensitivity(kept, "eye")
        if SS.get("fixed_t0") is not None and len(SS["fixed_t0"]):
            show_cached("fixedt0:" + df_key(SS["fixed_t0"]),
                        lambda: [("fixed t0", figures.fixed_t0_plot(SS["fixed_t0"], "eye"))])
        if st.button("Identifiability sweep (saccades)", width="stretch"):
            with st.spinner("Refitting every saccade cell at floors of 40–90 ms — "
                            "this is six full fits per cell, so give it a few minutes…"):
                SS["ident"] = analysis.identifiability_sweep(kept, "eye")
        if SS.get("ident") is not None and len(SS["ident"]):
            show_cached("ident:" + df_key(SS["ident"]),
                        lambda: [("identifiability", figures.identifiability_plot(SS["ident"], "eye"))])
        if st.button("Vincentiles (RT distributions)", width="stretch"):
            SS["vinc"] = True
        if SS.get("vinc"):
            show_cached("vincentiles", lambda: figures.vincentile_suite(kept))


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
        if st.button("Compare estimated vs fixed t₀ (LOO)", width="stretch"):
            box = st.status("Fitting both models for LOO…", expanded=True)
            try:
                SS["loo"] = compare.compare_t0_modes(kept, eff_cmp, draws=500, tune=500,
                                                     chains=2, status=box.write)
                box.update(label="LOO comparison complete", state="complete")
            except Exception as e:
                box.update(label="LOO failed", state="error"); st.error(str(e))
        if SS.get("loo"):
            st.markdown(f"Preferred model: **{SS['loo']['preferred']}**")
            st.dataframe(SS["loo"]["table"].round(2), width="stretch", hide_index=True)
            st.caption(SS["loo"]["note"])
    with c2:
        ui.eyebrow("Frequentist check")
        if st.button("Method A (differential evolution)", width="stretch"):
            box = st.status(f"Frequentist MLE for {eff_cmp}…", expanded=True)
            try:
                SS["freq"] = frequentist.fit_ddm(kept, eff_cmp, status=box.write)
                box.update(label="Frequentist fit complete", state="complete")
            except Exception as e:
                box.update(label="Frequentist fit failed", state="error"); st.error(str(e))
        if SS.get("freq"):
            st.caption(f"Method A group parameters — {eff_cmp}")
            st.dataframe(SS["freq"]["group"].round(2), width="stretch", hide_index=True)

    st.divider()
    ui.eyebrow("Per-speed hierarchical model (group parameters with credible intervals)")
    ui.hint("Treats speed as a modelled factor with participant random effects, so you get "
            "group-level v, a, and t₀ per speed <b>with</b> uncertainty — and optionally "
            "correlated participant effects (LKJ).")
    correlated = st.toggle("Model correlated participant effects (LKJ)", value=False)
    if st.button("Fit per-speed hierarchical model", width="stretch"):
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
        st.dataframe(ps["group"].round(2), width="stretch", hide_index=True)
        show_fig(figures.group_ci_plot(ps["group"], ps["effector"], "t0_ms"))
        if ps["corr"] is not None:
            st.caption("Participant-effect correlation matrix (LKJ)")
            st.dataframe(ps["corr"], width="stretch")
            ui.hint("Off-diagonal terms show how participants' parameters covary — "
                    "structure the independent-effects model cannot represent.")


# --------------------------------------------------------------------------- Step 4
if SS.results or SS.later:
    with st.container(border=True):
        ui.section("Results & report", "Parameters, diagnostics, graphs, advanced analyses, "
                   "and a downloadable report.", "4")
        kept = SS.filtered; res_all = SS.results
        # A selector rather than st.tabs, for two reasons. st.tabs resets to its first
        # tab whenever anything on the page triggers a rerun, so pressing a button in
        # the Download view threw you back to Parameters. And st.tabs renders every
        # panel on every run whether or not it is showing, so the eleven figures in
        # Graphs were redrawn each time any control moved. Keying the choice into
        # session state keeps the view put, and only the chosen panel is built.
        VIEWS = ["Parameters", "Diagnostics", "Graphs", "LATER",
                 "Advanced", "Model comparison", "Download"]
        st.segmented_control("Results view", VIEWS, key="sec4_view",
                             default=VIEWS[0], label_visibility="collapsed")
        view = SS.get("sec4_view") or VIEWS[0]

        # -------------------------------------------------- Parameters
        if view == "Parameters":
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
                    st.dataframe(r["group"].round(2), width="stretch", hide_index=True)
                if isinstance(r.get("units"), pd.DataFrame) and len(r["units"]):
                    with st.expander(f"{eff} — per-participant × speed estimates"):
                        st.dataframe(r["units"], width="stretch", hide_index=True)
                if isinstance(r.get("mixture"), pd.DataFrame) and len(r["mixture"]):
                    ui.eyebrow("Two-component saccade cells")
                    mixtbl = r["mixture"].copy()
                    if "express_mode" in mixtbl.columns:
                        # a fast component is only an express saccade if it is actually
                        # under the 130 ms express cutoff -- usually it is not
                        mixtbl["fast mode < 130 ms"] = mixtbl["express_mode"] < 130
                    st.dataframe(mixtbl[[c for c in ["participant", "speed", "n", "pi",
                                                     "express_mode", "reg_mode",
                                                     "fast mode < 130 ms"]
                                         if c in mixtbl.columns]].round(2),
                                 width="stretch", hide_index=True)
                    n_true = int((mixtbl["express_mode"] < 130).sum()) if "express_mode" in mixtbl else 0
                    ui.hint(f"{len(mixtbl)} cells needed two components; in {n_true} of them the "
                            f"faster component is actually in express territory (under 130 ms). "
                            f"The rest are two ordinary-latency modes, so these are not "
                            f"express-saccade participants — see the LATER tab for that.")
            if not any_p:
                st.info("Run a fit to see parameters.")

        # -------------------------------------------------- Diagnostics
        if view == "Diagnostics":
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
                    st.dataframe(g["by_condition"].round(3), width="stretch", hide_index=True)
            if not shown:
                st.info("Run the full Bayesian fit to see convergence and goodness-of-fit.")

        # -------------------------------------------------- Graphs
        if view == "Graphs":
            ui.hint("Visual model checks and summaries, in the repository's house style. "
                    "Figures are drawn once per fit and kept, so moving around the app "
                    "does not redraw them.")

            def graph_figs():
                out = []
                for eff in EFFECTORS:
                    r = res_all.get(eff)
                    if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                        out += figures.ddm_schematic_figs(kept, r["group"], eff)
                for eff in EFFECTORS:
                    r = res_all.get(eff)
                    if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                        out.append((f"{eff.capitalize()} fit",
                                    figures.fit_overlay(kept, eff, r["group"])))
                if res_all:
                    out.append(("Non-decision time by speed", figures.ndt_by_speed(res_all)))
                out.append(("Why saccadic t0 floors", figures.why_floors(kept)))
                if SS.later is not None:
                    out.append(("LATER reciprobit",
                                figures.reciprobit(SS.later, kept[kept.effector == "eye"])))
                return out

            show_cached("graphs", graph_figs)

        if view == "LATER":
            if SS.later is not None:
                lat = SS.later
                cA, cB = st.columns(2)
                cA.metric("Median reciprobit R²", f"{lat['median_r2']:.2f}")
                cB.metric("Express-dominant participants",
                          f"{int(lat['per_participant']['express_dominant'].sum())} / "
                          f"{len(lat['per_participant'])}")
                st.dataframe(lat["per_participant"].round(3), width="stretch", hide_index=True)
            else:
                st.info("LATER runs when saccadic (eye) trials are included.")

        # -------------------------------------------------- Advanced / Comparison (fragments)
        if view == "Advanced":
            advanced_tab()
        if view == "Model comparison":
            comparison_tab()

        # -------------------------------------------------- Download
        if view == "Download":
            ui.eyebrow("Report & figures")
            ui.hint("HTML report to read in a browser · a vector PDF of every figure · or a full "
                    "ZIP with the report, figures (PNG + PDF), and all tables as CSV. "
                    "Building redraws every figure, so it is done when you ask for it rather "
                    "than on every interaction.")

            def build_context(bar):
                """Redraw every figure for the report, reporting progress as it goes."""
                ctx = {"title": "KINARM RT analysis report",
                       "subtitle": "Generated by the KINARM RT app",
                       "filter_report": SS.filter_report,
                       "cell_summary": data.cell_summary(kept),
                       "results": res_all,
                       "gof": {e: res_all[e].get("gof") for e in res_all if res_all[e].get("gof")},
                       "later": SS.later, "figures": {}}
                # each entry is (label, thunk) so the work is counted before it starts
                jobs = []
                for eff in EFFECTORS:
                    r = res_all.get(eff)
                    if r and isinstance(r.get("group"), pd.DataFrame) and len(r["group"]):
                        jobs.append((f"{eff} schematics",
                                     lambda e=eff, g=r["group"]: figures.ddm_schematic_figs(kept, g, e)))
                        jobs.append((f"{eff.capitalize()} fit",
                                     lambda e=eff, g=r["group"]: figures.fit_overlay(kept, e, g)))
                if res_all:
                    jobs.append(("Non-decision time by speed", lambda: figures.ndt_by_speed(res_all)))
                jobs.append(("Why saccadic t0 floors", lambda: figures.why_floors(kept)))
                jobs.append(("RT distributions", lambda: figures.vincentile_suite(kept)))
                if SS.later is not None:
                    jobs.append(("LATER reciprobit",
                                 lambda: figures.reciprobit(SS.later, kept[kept.effector == "eye"])))

                for i, (label, thunk) in enumerate(jobs, 1):
                    bar.note(f"drawing {label}")
                    out = thunk()
                    if isinstance(out, list):                 # a suite of figures
                        for lab, f in out:
                            ctx["figures"][lab] = f
                    else:
                        ctx["figures"][label] = out
                    bar(i, len(jobs))
                return ctx

            ARTEFACTS = {
                "html": ("HTML report", "kinarm_rt_report.html", "text/html",
                         lambda c: report.build_html_report(c)),
                "pdf": ("Figures (PDF)", "kinarm_rt_figures.pdf", "application/pdf",
                        lambda c: report.build_figures_pdf(c["figures"])),
                "zip": ("Full bundle (ZIP)", "kinarm_rt_results.zip", "application/zip",
                        lambda c: report.build_zip_bundle(c)),
            }

            # Built in place, without st.rerun(). A rerun restarts the script from the
            # top, which resets the tab strip to its first tab -- so asking for a
            # download used to throw you out of this tab and back into Parameters.
            cols = st.columns(3)
            asked = None
            for col, key in zip(cols, ARTEFACTS):
                label, fname, mime, _builder = ARTEFACTS[key]
                ready = SS.get(f"dl_{key}")
                if ready is not None:
                    col.download_button(f"Download {label}", ready, file_name=fname,
                                        mime=mime, width="stretch", key=f"dlb_{key}")
                elif col.button(f"Prepare {label}", width="stretch", key=f"prep_{key}"):
                    asked = key

            if asked:
                label, fname, mime, builder = ARTEFACTS[asked]
                panel = st.status(f"Building {label}…", expanded=True)
                bar = ui.StepBar(panel, "Figures", unit="figures")
                try:
                    ctx = build_context(bar)
                    bar.finish(f"{len(ctx['figures'])} figures")
                    wbar = ui.StepBar(panel, f"Assembling {label}", unit="steps")
                    wbar.note("writing file")
                    blob = builder(ctx)
                    SS[f"dl_{asked}"] = blob
                    wbar.finish(f"{len(blob) / 1e6:.1f} MB")
                    panel.update(label=f"{label} ready", state="complete", expanded=False)
                    # offered right here, so nothing moves and no rerun is needed
                    st.download_button(f"Download {label}", blob, file_name=fname, mime=mime,
                                       width="stretch", key=f"dlnow_{asked}",
                                       type="primary")
                except Exception as e:
                    panel.update(label=f"{label} failed", state="error")
                    st.error(f"Could not build {label}: {e}")

            try:
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
                                            width="stretch")
            except Exception as e:
                st.error(f"Report error: {e}")
