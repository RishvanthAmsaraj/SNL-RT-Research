"""
KINARM RT -- point-and-click front end for the SNL-RT-Research reaction-time pipelines.

The app does not re-implement any analysis. It runs the repository's own scripts (vendored unchanged in
pipelines/, checked against MANIFEST.json) and shows what they write, with an explanation next to everything.

Navigation rule: session-state keys that belong to widgets are only ever changed in callbacks (on_click /
on_change), never after the widget has been drawn -- Streamlit forbids that and raises.
"""
from __future__ import annotations
import hashlib, json, os, re, tempfile, time, traceback
from collections import Counter
from dataclasses import dataclass
import streamlit as st

from kinarm_rt import __version__, theme, explain as X, insights as I
from kinarm_rt.engine import registry as R, runner, intake, results

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="KINARM RT", page_icon="◎", layout="wide", initial_sidebar_state="collapsed")
theme.inject(theme.current_theme())
ss = st.session_state
TOP = ["Experiment 1", "Experiment 2", "Compare"]
VIEWS = ["Overview", "Run", "Figures", "Tables", "Documents", "Models"]
CVIEWS = ["Side by side", "Figures", "Tables", "Documents"]
EXP_IDS = {"Experiment 1": "E1", "Experiment 2": "E2"}
for k, v in {"exp_w": "Experiment 1", "exp_last": "Experiment 1", "view_w": "Overview", "view_last": "Overview",
             "cview_w": "Side by side", "cview_last": "Side by side", "fig": None, "runs": {}, "source": {},
             "chosen": {}, "intake": {}, "sbs": {}}.items():
    ss.setdefault(k, v)

if not ss.get("_qp_done"):                         # deep links (?exp=E2&view=Figures&fig=...), applied before any widget
    qp = st.query_params
    top0 = {"E1": "Experiment 1", "E2": "Experiment 2", "compare": "Compare"}.get(qp.get("exp", ""))
    if top0: ss.exp_w = ss.exp_last = top0
    if qp.get("view") in VIEWS: ss.view_w = ss.view_last = qp.get("view")
    if qp.get("cview") in CVIEWS: ss.cview_w = ss.cview_last = qp.get("cview")
    if qp.get("fig"): ss.fig = qp.get("fig")
    ss._qp_done = True


# --------------------------------------------------------------------------- navigation (callbacks only)
def keep(widget_key: str, last_key: str) -> None:
    """A segmented control can be clicked off; put the previous choice back so one is always selected."""
    if ss.get(widget_key) is None: ss[widget_key] = ss[last_key]
    else: ss[last_key] = ss[widget_key]
    ss.fig = None


def nav(top=None, view=None, cview=None, fig="__keep__", exp_id=None, src=None, cats=None) -> None:
    if top: ss.exp_w = ss.exp_last = top
    if view: ss.view_w = ss.view_last = view
    if cview: ss.cview_w = ss.cview_last = cview
    if fig != "__keep__": ss.fig = fig
    if exp_id and src: ss.source[exp_id] = src; ss[f"src_{exp_id}"] = src
    if exp_id and cats is not None: ss[f"cats_{exp_id}"] = cats; ss[f"q_{exp_id}"] = ""


def _reset_pane(side: str) -> None:
    for k in (f"sbs_src_{side}", f"sbs_fig_{side}"): ss.pop(k, None)     # re-initialised from ss.sbs on the next run


def pin(side: str, source_id: str, stem: str) -> None:
    """Open a figure side by side. If the other experiment has the same figure, show both experiments' versions
    (Experiment 1 on the left); otherwise put the figure in the given pane and leave the other one as it was."""
    kind = source_id.split(":")[0]
    if kind in ("E1", "E2"):
        other = "E2" if kind == "E1" else "E1"
        if stem in results.reference(other).image_stems():
            left = source_id if kind == "E1" else "E1:ref"; right = source_id if kind == "E2" else "E2:ref"
            ss.sbs["L"] = (left, stem); ss.sbs["R"] = (right, stem); _reset_pane("L"); _reset_pane("R")
            nav(top="Compare", cview="Side by side"); return
    ss.sbs[side] = (source_id, stem); _reset_pane(side)
    nav(top="Compare", cview="Side by side")


def preset(stem: str) -> None:
    for side, src in (("L", "E1:ref"), ("R", "E2:ref")):
        ss.sbs[side] = (src, stem); _reset_pane(side)


# --------------------------------------------------------------------------- small helpers
def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def show_image(path: str) -> None:
    """No width argument on purpose: in Streamlit 1.50, width="stretch" and use_container_width both draw an image 16 px
    wide inside a column. Without one, every version draws it at its own size, capped to the container (our images are
    always at least as wide as their container)."""
    st.image(path)


def row(key: str):
    """Buttons side by side with one consistent gap (falls back to a plain container on old Streamlit)."""
    try: return st.container(horizontal=True, gap="small", key=f"row-{key}")
    except TypeError: return st.container(key=f"row-{key}")


def show_df(df) -> None:
    try: st.dataframe(df, width="stretch", hide_index=True)
    except TypeError: st.dataframe(df, use_container_width=True, hide_index=True)


def mime(name: str) -> str:
    return {"png": "image/png", "pdf": "application/pdf", "csv": "text/csv", "zip": "application/zip"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")


def guarded(fn, *args, **kw) -> None:
    """Draw one section; if it fails, say so in place instead of taking the whole app down."""
    try:
        fn(*args, **kw)
    except Exception as exc:
        if type(exc).__name__ in ("RerunException", "StopException", "RerunData"): raise
        with st.container(key=f"kd-err-{fn.__name__}-{abs(hash(str(exc))) % 10**6}"):
            md(f"<b>This section could not be drawn.</b> {type(exc).__name__}: {exc}")
            with st.expander("Technical details"): st.code(traceback.format_exc()[-4000:])


@st.cache_data(show_spinner=False)
def _zip(folders: tuple, stamp: float) -> bytes:
    return results.ResultSet("", "", folders).zip_bytes()


@st.cache_data(show_spinner=False, ttl=3600)
def env_summary() -> str:
    e = runner.environment()
    return ", ".join(f"{k} {v}" for k, v in e.items() if k in ("python", "numpy", "scipy", "pymc"))


@st.cache_data(show_spinner=False, ttl=3600)
def sync_status() -> tuple:
    path = os.path.join(HERE, "pipelines", "MANIFEST.json")
    try: man = json.load(open(path))
    except (OSError, ValueError): return 0, -1
    ok = bad = 0
    for name, files in man["pipelines"].items():
        for rec in files:
            p = os.path.join(HERE, "pipelines", name, rec["file"])
            if os.path.exists(p) and runner.file_sha(p) == rec["sha256"]: ok += 1
            else: bad += 1
    return ok, bad


def run_dir_for(exp_id: str):
    d = ss.runs.get(exp_id)
    return d if d and os.path.isdir(d) else None


def result_set(exp: R.Experiment) -> results.ResultSet:
    d = run_dir_for(exp.id)
    if ss.source.get(exp.id) == "Your run" and d: return results.from_run(d)
    return results.reference(exp.id)


def source_by_id(source_id: str):
    kind, which = source_id.split(":")
    if kind == "CMP": return None, results.compare_reference()
    exp = R.EXPERIMENTS[kind]
    if which == "run" and run_dir_for(kind): return exp, results.from_run(run_dir_for(kind))
    return exp, results.reference(kind)


def exp_of(source_id: str):
    kind = source_id.split(":")[0]
    return R.EXPERIMENTS.get(kind)


def source_options() -> dict:
    opts = {"E1:ref": "Experiment 1, lab results", "E2:ref": "Experiment 2, lab results"}
    for e in ("E1", "E2"):
        if run_dir_for(e): opts[f"{e}:run"] = f"{R.EXPERIMENTS[e].name}, your run"
    opts["CMP:ref"] = "Comparison figures"
    return opts


# --------------------------------------------------------------------------- figures as items
@dataclass
class Item:
    stem: str
    title: str
    category: str
    fig: object


ARCH_RE = re.compile(r"^(v1|v2|v2_5|v3|wi)__(.+)$")


def catalogue_match(base: str):
    """The catalogue entry (and speed) describing a figure file stem, matching per-speed names too."""
    for f in X.FIGURES + X.COMPARE_FIGURES:
        if f.per_speed:
            m = re.fullmatch(re.escape(f.stem).replace(r"\{s\}", r"(\d+)"), base)
            if m: return f, int(m.group(1))
        elif f.stem == base:
            return f, None
    return None, None


def archive_label(name: str) -> str:
    """'v3__ddm_hrt_75_degs' -> 'Model schematic, hand (Method A), 75 deg/s (pipeline version 3)'."""
    m = ARCH_RE.match(name)
    if not m: return name
    tag, rest = m.group(1), m.group(2); parts = rest.split("__"); base = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
    f, spd = catalogue_match(base)
    title = (f.title + (f", {spd} deg/s" if spd is not None else "")) if f else base.replace("_", " ")
    title = title[:1].upper() + title[1:]
    extra = []
    if "demo" in rest.lower(): extra.append("synthetic demo data")
    elif len(parts) > 1 and tag != "wi": extra.append(parts[0].replace("-", " ").lower())
    return f"{title} ({X.ARCHIVE_LABEL.get(tag, tag)}{', ' + ', '.join(extra) if extra else ''})"


def archive_item(stem: str) -> Item:
    tag = ARCH_RE.match(stem).group(1); label = X.ARCHIVE_LABEL.get(tag, tag)
    f, _ = catalogue_match(stem.split("__")[-1]); title = archive_label(stem)
    fig = X.Fig(stem, title, "deprecated", "archive",
                f"The {label} copy of this figure" + (f". {f.what}" if f else ", as it was produced at the time."),
                f.read if f else "Read it as it was produced at the time; the current pipeline's version of the figure, where there is one, is the reference.",
                "Superseded by the current pipeline and kept for the record. It either shows a mistake that has since been "
                "corrected or comes from an earlier fit, so its floors, labels and numbers can differ from the current figure; use "
                "the current figure for conclusions.",
                why=f"It is part of the project's history: the {label} version of a figure that has since been replaced.")
    return Item(stem, title, "deprecated", fig)


def figure_items(exp, rs: results.ResultSet) -> list:
    """One item per figure that exists in these results: catalogued figures first (per-speed ones split by speed),
    then anything the catalogue does not describe, so no figure is ever hidden."""
    have = set(rs.image_stems()); out, seen = [], set()
    cat = X.COMPARE_FIGURES if exp is None else [f for f in X.FIGURES if exp.id in f.experiments]
    for f in cat:
        speeds = exp.speeds if (exp and f.per_speed) else (None,)
        for s in speeds:
            stem = f.stem.format(s=s) if s is not None else f.stem
            if stem in have and stem not in seen:
                out.append(Item(stem, f.title + (f", {s} deg/s" if s is not None else ""), f.category, f)); seen.add(stem)
    order = {t: i for i, t in enumerate(("v3", "v2_5", "v2", "v1", "wi"))}
    rest = sorted(have - seen, key=lambda st_: (order.get(st_.split("__")[0], 9), st_))
    for stem in rest:
        out.append(archive_item(stem) if ARCH_RE.match(stem) else Item(stem, stem.replace("_", " "), "other", None))
    return out


def missing_items(exp, rs) -> list:
    if exp is None: return []
    have = set(rs.image_stems()); out = []
    for f in X.FIGURES:
        if exp.id not in f.experiments: continue
        stems = [f.stem.format(s=s) for s in exp.speeds] if f.per_speed else [f.stem]
        if not any(s in have for s in stems): out.append(f)
    return out


# --------------------------------------------------------------------------- header and hero
def header() -> None:
    h1, h2 = st.columns([1, 1.3], vertical_alignment="center")
    with h1:
        md(f'<div class="k-brand">◎ KINARM RT <small>reaction-time models, version {__version__}</small></div>')
    with h2:
        st.segmented_control("Experiment", TOP, key="exp_w", label_visibility="collapsed", on_change=keep, args=("exp_w", "exp_last"))


def hero_for(exp: R.Experiment, rs: results.ResultSet) -> None:
    hl = results.headline(rs, exp.speeds)
    chips = []
    if "participants" in hl: chips.append(("Participants", str(hl["participants"])))
    chips.append(("Speeds", " / ".join(str(s) for s in exp.speeds) + " deg/s"))
    if "hand_t0" in hl: chips.append(("Hand t₀", " / ".join(f"{v:.0f}" for v in hl["hand_t0"].values()) + " ms"))
    if "sacc" in hl: chips.append(("Saccadic t₀", "fixed at 70 ms"))
    body = ("Participants intercept a target on a ring. It is stationary or travels counter-clockwise at 75 or 150 deg/s. "
            "Hand and eye reaction times are each split into non-decision time and decision time with a single-boundary shifted Wald."
            if exp.id == "E1" else
            "The same interception task with every target moving, at 75, 100, 125 or 150 deg/s, analysed with the same "
            "method as Experiment 1 in its own copy of the code.")
    theme.hero(f"{exp.name}, {exp.cohort} cohort", exp.accent, exp.tagline, body, chips, exp.speeds, R.SPEED_COLOURS)


def source_switch(exp: R.Experiment) -> None:
    opts = ["Lab results", "Your run"] if run_dir_for(exp.id) else ["Lab results"]
    key = f"src_{exp.id}"
    if ss.get(key) not in opts: ss[key] = ss.source.get(exp.id) if ss.source.get(exp.id) in opts else "Lab results"
    c1, c2 = st.columns([1.3, 3], vertical_alignment="center")
    with c1: st.segmented_control("Results shown", opts, key=key, label_visibility="collapsed")
    ss.source[exp.id] = ss.get(key) or "Lab results"
    with c2:
        md('<span class="k-muted">' + ("The lab's committed results, made by these same scripts. Nothing needs to run."
                                        if ss.source[exp.id] == "Lab results" else "Results from your own run.") + "</span>")


# --------------------------------------------------------------------------- sections
def overview(exp: R.Experiment) -> None:
    source_switch(exp); rs = result_set(exp); hl = results.headline(rs, exp.speeds)
    with st.container(key=f"kp-ov-{exp.id}"):
        md("<h3>What the results say</h3>")
        if "hand_t0" in hl:
            vals = " / ".join(f"{v:.0f}" for v in hl["hand_t0"].values()); p = hl.get("friedman_p")
            md(f'<p class="k-sub"><b>Hand non-decision time</b> is {vals} ms at {" / ".join(map(str, exp.speeds))} deg/s, '
               f'with {hl["hand_floored"]} of {hl["hand_cells"]} cells on the 130 ms floor'
               + (f" (Friedman p = {p:.3f} across speeds)." if p is not None else ".") + "</p>")
        else:
            md('<p class="k-sub">There are no Bayesian hand fits in these results yet. Run the hierarchical Bayesian analysis to make them.</p>')
        if "sacc" in hl:
            s = hl["sacc"]
            md(f'<p class="k-sub"><b>Saccadic non-decision time</b> is not identifiable. Of {s["n"]} participants, {s["floor"]} have '
               f'an interval on the 70 ms floor, {s["ceiling"]} on the ceiling set by their fastest saccade, and {s["free"]} '
               + ("sits" if s["free"] == 1 else "sit") + " clear of both, so it is reported as fixed at 70 ms.</p>")
        if "later_r2" in hl:
            md(f'<p class="k-sub"><b>LATER</b>, a deprecated model: median reciprobit r² = {hl["later_r2"]:.3f}.</p>')
        with row(f"ov-{exp.id}"):
            st.button("See all figures", type="primary", key=f"go_fig_{exp.id}", on_click=nav, kwargs=dict(view="Figures", fig=None))
            st.button("Run on your data", key=f"go_run_{exp.id}", on_click=nav, kwargs=dict(view="Run"))
            st.button("Compare experiments", key=f"go_cmp_{exp.id}", on_click=nav, kwargs=dict(top="Compare", cview="Side by side"))
    items = figure_items(exp, rs)
    with st.container(key=f"kp-all-{exp.id}"):
        md("<h3>Everything in the repository, in one place</h3>")
        md('<div class="k-chips">' + "".join(f'<span class="k-chip">{k} <b>{v}</b></span>' for k, v in (
            ("Figures", f"{len(items)}"), ("Figure files", f"{results.original_figure_files(rs)} PNG and PDF"),
            ("Tables", f"{len(rs.tables())}"), ("Documents", f"{len(results.documents(exp.id))}"))) + "</div>")
        md('<p class="k-sub" style="margin-top:8px">Each figure is listed once and offers both of its files, PNG and PDF. '
           "Older figures from earlier pipeline versions, and the deprecated models, are under Deprecated.</p>" if exp.id == "E1" else
           '<p class="k-sub" style="margin-top:8px">Each figure is listed once and offers both of its files, PNG and PDF.</p>')
    with st.container(key=f"kp-key-{exp.id}"):
        md("<h3>Key figures</h3>")
        cols = st.columns(3)
        for col, stem in zip(cols, ("NDT_barchart_bayesian", "HRT_floor_control", "why_saccadic_t0_floors")):
            f = next(x for x in X.FIGURES if x.stem == stem)
            with col, st.container(key=f"kq-card-key-{exp.id}-{stem}"):
                img = rs.thumb(stem)
                if img: show_image(img)
                md(f'<div class="k-title">{f.title}</div><div class="k-row">{theme.badge(f.category, X.CATEGORY_LABEL[f.category])}</div>')
                st.button("Open", key=f"key_open_{exp.id}_{stem}", on_click=nav, kwargs=dict(view="Figures", fig=stem), disabled=not img)
    with st.container(key=f"kp-how-{exp.id}"):
        md("<h3>How this app works</h3>")
        md('<p class="k-sub">Every result comes from the repository\'s own scripts, run unchanged: the same files the lab runs by hand, '
           "in the same order, with the same settings. The app chooses which scripts to run, tracks their progress, and explains each "
           "figure and table. Deprecated models stay available, labelled, with the reason they were set aside.</p>")


def models_view(exp: R.Experiment) -> None:
    cats_for = {"bayesian": ["Main"], "identifiability": ["Diagnostic"], "model_free": ["Main"],
                "method_a": ["Method A"], "later": ["Deprecated"], "two_boundary": ["Deprecated"]}
    for status in R.STATUS_ORDER:
        for a in [a for a in R.analyses_for(exp.id) if a.status == status]:
            note = X.MODEL_NOTES.get(a.id)
            with st.container(key=f"{'kd' if status == 'deprecated' else 'kp'}-model-{exp.id}-{a.id}"):
                md(f"<h3>{a.name} {theme.badge(a.status, R.STATUS_LABEL[a.status])}</h3>")
                md(f'<p class="k-sub">{a.summary}</p>')
                if note:
                    md(f'<p class="k-sub"><b>What it is.</b> {note.what}</p><p class="k-sub"><b>'
                       + ("Why it is deprecated." if a.status == "deprecated" else "Why it is here.") + f"</b> {note.why}</p>")
                    if note.evidence: md("<ul class='k-sub'>" + "".join(f"<li>{e}</li>" for e in note.evidence) + "</ul>")
                    if note.still_useful: md(f'<p class="k-muted">Still useful for {note.still_useful[0].lower() + note.still_useful[1:]}</p>')
                if a.id in cats_for:
                    st.button("See its figures", key=f"mfig_{exp.id}_{a.id}", on_click=nav,
                              kwargs=dict(view="Figures", fig=None, exp_id=exp.id, cats=cats_for[a.id],
                                          src="Lab results" if a.reference_only else None))


# ---- run
def data_panel(exp: R.Experiment) -> None:
    with st.container(key=f"kp-data-{exp.id}"):
        md("<h3>Your data</h3>")
        hint = ("A pooled trial file in the pipeline's format, like pooled_data.csv." if exp.id == "E1" else
                "A pooled trial file like pooled_data_P2.csv, or all the per-participant CIR…_TRIAL_Summary files at once; the app "
                "then builds the pooled file with the repository's builder script.")
        md(f'<p class="k-sub">{hint} Required columns: {", ".join(intake.REQUIRED)}.</p>')
        files = st.file_uploader("Data files", type=["csv"], accept_multiple_files=True, key=f"up_{exp.id}", label_visibility="collapsed")
        sample = os.path.join(HERE, "sample_data", "example_pooled_data.csv")
        use_sample = (exp.id == "E1" and not files and os.path.exists(sample)
                      and st.toggle("Use the bundled sample data (synthetic, for trying the app)", key="sample_E1"))
        if not files and not use_sample: return
        payload = [("example_pooled_data.csv", open(sample, "rb").read())] if use_sample else [(f.name, f.getvalue()) for f in files]
        sig = hashlib.sha256(b"".join(n.encode() + hashlib.sha256(b).digest() for n, b in payload)).hexdigest()
        cached = ss.intake.get(exp.id)
        if not cached or cached.get("sig") != sig:
            try:
                with st.spinner("Checking the data…"):
                    if len(payload) > 1 and exp.raw_builder:
                        pooled, audit = intake.build_e2_from_raw(payload, exp)
                    else:
                        pooled = os.path.join(tempfile.mkdtemp(prefix="kinarm_in_"), exp.data_file)
                        with open(pooled, "wb") as out: out.write(payload[0][1])
                        audit = ""
                    rep = intake.validate(intake.read_table(pooled), exp)
                ss.intake[exp.id] = {"sig": sig, "pooled": pooled, "audit": audit, "report": rep}
            except Exception as exc:
                ss.intake.pop(exp.id, None); st.error(f"The data could not be read: {exc}"); return
        info = ss.intake[exp.id]; rep = info["report"]
        for e in rep.errors: st.error(e)
        for w in rep.warnings: st.warning(w)
        if rep.facts:
            md('<div class="k-chips">' + "".join(f'<span class="k-chip">{k.capitalize()} <b>{v:,}</b></span>' for k, v in rep.facts.items()) + "</div>")
        if info["audit"]:
            with st.expander("Builder audit"): st.code(info["audit"][:6000])
        if rep.ok:
            try:
                ss.runs[exp.id] = runner.stage(exp, info["pooled"])
                md('<p class="k-sub" style="color:var(--k-rec)"><b>Data checked.</b> Choose what to run below.</p>')
            except OSError as exc:
                st.error(f"The run folder could not be created in {runner.runs_root()}: {exc}")


def analysis_picker(exp: R.Experiment) -> list:
    chosen = ss.chosen.setdefault(exp.id, {"bayesian", "identifiability", "model_free"})
    ok_bayes, why = runner.bayes_ready()
    with st.container(key=f"kp-pick-{exp.id}"):
        md("<h3>What to run</h3>")
        md(f'<p class="k-muted">Bayesian fits: {"ready." if ok_bayes else why + "."} Steps already run on the same data are reused.</p>')
        for a in R.analyses_for(exp.id):
            box = "kd" if a.status == "deprecated" else "kq"
            with st.container(key=f"{box}-pick-{exp.id}-{a.id}"):
                c1, c2 = st.columns([5, 1.4], vertical_alignment="center")
                with c2:
                    if a.reference_only:
                        st.button("View results", key=f"see_{exp.id}_{a.id}", on_click=nav,
                                  kwargs=dict(view="Figures", fig=None, exp_id=exp.id, src="Lab results", cats=["Deprecated"]))
                    else:
                        on = st.toggle("Include", value=a.id in chosen, key=f"tg_{exp.id}_{a.id}")
                        (chosen.add if on else chosen.discard)(a.id)
                with c1:
                    md(f'<div class="k-row"><b>{a.name}</b>{theme.badge(a.status, R.STATUS_LABEL[a.status])}<span class="k-count">{a.minutes}</span></div>'
                       f'<div class="k-sub" style="margin:6px 0 0">{a.summary}</div>'
                       + (f'<div class="k-muted" style="margin-top:6px">{X.MODEL_NOTES[a.id].why}</div>'
                          if a.status == "deprecated" and (a.id in chosen or a.reference_only) else ""))
    return sorted(chosen)


ICON = {"done": "OK", "running": "↻", "failed": "X", "skipped": "–", "blocked": "–", "cancelled": "–"}


def _progress_body(exp: R.Experiment, run_dir: str) -> bool:
    state = runner.load_state(run_dir); steps = R.steps_for(exp); planned = state.get("plan", [])
    if not planned: return False
    running = runner.is_running(run_dir)
    total = sum(steps[s].minutes for s in planned if s in steps) or 1
    finished = sum(steps[s].minutes for s in planned if s in steps and state["steps"].get(s, {}).get("status") not in (None, "running"))
    rows = []
    for sid in planned:
        rec = state["steps"].get(sid, {}); status = rec.get("status") or "pending"
        if status == "running": t = f"{time.time() - rec.get('started', time.time()):.0f} s"
        elif status == "done" and rec.get("cached"): t = "reused"
        elif status == "done": t = f"{rec.get('ended', 0) - rec.get('started', 0):.0f} s"
        else: t = ""
        det = f'<div class="detail">{rec["detail"]}</div>' if rec.get("detail") else ""
        rows.append(f'<li class="{status}"><span class="s">{ICON.get(status, "○")}</span><span>{steps[sid].label if sid in steps else sid}{det}</span>'
                    f'<span class="t">{t}</span></li>')
    with st.container(key=f"kp-prog-{exp.id}"):
        overall = state.get("status", "")
        md("<h3>" + ("Running" if running else (f"Run {overall}" if overall and overall != "idle" else "Run")) + "</h3>")
        st.progress(min(finished / total, 1.0) if running else 1.0)
        md('<ul class="k-steps">' + "".join(rows) + "</ul>")
        cur = next((s for s in planned if state["steps"].get(s, {}).get("status") == "running"), None)
        bad = [s for s in planned if state["steps"].get(s, {}).get("status") == "failed"]
        for sid in ([cur] if cur else []) + bad:
            with st.expander(f"Log: {steps[sid].label}", expanded=bool(bad)): st.code(runner.log_tail(run_dir, sid) or "(no output yet)")
        with row(f"prog-{exp.id}"):
            if running:
                st.button("Cancel run", key=f"cancel_{exp.id}", on_click=runner.cancel, args=(run_dir,))
            else:
                st.button("See figures", type="primary", key=f"after_fig_{exp.id}", on_click=nav,
                          kwargs=dict(view="Figures", fig=None, exp_id=exp.id, src="Your run"))
                stamp = max((os.path.getmtime(os.path.join(run_dir, f)) for f in os.listdir(run_dir)), default=0)
                st.download_button("Download results (zip)", _zip((run_dir,), stamp), file_name=f"kinarm_rt_{exp.id}_results.zip",
                                   mime="application/zip", key=f"dl_run_{exp.id}")
        if not running:
            for a in R.analyses_for(exp.id):
                if a.status == "deprecated" and a.steps and all(state["steps"].get(s, {}).get("status") == "done" for s in a.steps):
                    hl = results.headline(results.from_run(run_dir), exp.speeds)
                    with st.container(key=f"kd-after-{exp.id}-{a.id}"):
                        md(f"<b>You ran the {a.name}.</b> " + (f"Median reciprobit r² = {hl['later_r2']:.3f}. " if "later_r2" in hl else "")
                           + X.MODEL_NOTES[a.id].why)
    return running


def progress_panel(exp: R.Experiment, run_dir: str) -> None:
    if runner.is_running(run_dir):
        @st.fragment(run_every=2.0)
        def live_progress():
            if not _progress_body(exp, run_dir): st.rerun()
        live_progress()
    else:
        _progress_body(exp, run_dir)


def run_view(exp: R.Experiment) -> None:
    guarded(data_panel, exp)
    chosen = analysis_picker(exp)
    run_dir = run_dir_for(exp.id)
    if not run_dir:
        md('<p class="k-muted">Load a data file above to run anything. The lab\'s own results are already under Figures and Tables.</p>'); return
    if runner.interrupted(run_dir):
        st.info("The last run stopped before it finished, because the app was closed. Run again to continue; finished steps are reused.")
    steps = R.plan(exp, chosen); mins = sum(s.minutes for s in steps)
    info = ss.intake.get(exp.id); cols = info["report"].columns if info else set()
    missing = runner.missing_modules()
    if missing:
        st.error("This Python environment is missing " + ", ".join(missing) + ", which the pipeline scripts need. Install "
                 f"{'it' if len(missing) == 1 else 'them'} (pip install {' '.join(missing)}) or start the app from the conda "
                 "environment, then reload this page.")
    c1, c2 = st.columns([1.5, 4], vertical_alignment="center")
    c1.button("Run selected analyses", type="primary", disabled=runner.is_running(run_dir) or not steps or bool(missing),
              key=f"go_{exp.id}", on_click=runner.start, args=(exp, run_dir, steps, cols))
    when = "under a minute" if mins < 1 else f"about {mins:.0f} min"
    c2.markdown(f'<span class="k-muted">{len(steps)} step{"" if len(steps) == 1 else "s"}, {when} on a 4-core machine when nothing '
                "is reused.</span>", unsafe_allow_html=True)
    progress_panel(exp, run_dir)


# ---- figures
def figure_detail(item: Item, siblings: list, rs: results.ResultSet, key: str, source_id: str) -> None:
    stems = [i.stem for i in siblings]
    idx = stems.index(item.stem) if item.stem in stems else 0
    prev_stem = siblings[idx - 1].stem if siblings else item.stem
    next_stem = siblings[(idx + 1) % len(siblings)].stem if siblings else item.stem
    with st.container(key=f"kp-figd-{key}"):
        md(f"<h3>{item.title} {theme.badge(item.category, X.CATEGORY_LABEL[item.category])}</h3>")
        with row(f"figd-{key}"):
            st.button("Previous", key=f"prev_{key}", on_click=nav, kwargs=dict(fig=prev_stem), disabled=len(siblings) < 2)
            st.button("Next", key=f"next_{key}", on_click=nav, kwargs=dict(fig=next_stem), disabled=len(siblings) < 2)
            st.button("Side by side", key=f"pin_{key}", on_click=pin, args=("L", source_id, item.stem))
            st.button("Close", key=f"close_{key}", on_click=nav, kwargs=dict(fig=None))
            md(f'<span class="k-count">{idx + 1} of {len(siblings)}</span>')
        img = rs.image(item.stem)
        if img: show_image(img)
        else: st.info("A preview is not available for this figure; download the PDF below.")
        f = item.fig
        if f:
            facts = I.facts(item.stem, exp_of(source_id), rs)
            md(f'<p class="k-sub"><b>What it shows.</b> {f.what}</p>'
               + (f'<p class="k-sub"><b>Why it is here.</b> {f.why}</p>' if f.why else "")
               + f'<p class="k-sub"><b>How to read it.</b> {f.read}</p>'
               + (('<p class="k-sub" style="margin-bottom:2px"><b>In these results.</b></p><ul class="k-facts">'
                   + "".join(f"<li>{x}</li>" for x in facts) + "</ul>") if facts else "")
               + f'<p class="k-sub"><b>What it means for the project.</b> {f.meaning}</p>'
               + (f'<p class="k-muted"><b>Keep in mind.</b> {f.caveat}</p>' if f.caveat else ""))
            if f.category == "deprecated" and f.analysis in X.MODEL_NOTES:
                with st.container(key=f"kd-figdep-{key}"):
                    md(f"<b>Why this model is deprecated.</b> {X.MODEL_NOTES[f.analysis].why}")
        else:
            md('<p class="k-muted">This figure has no description in the app yet.</p>')
        with row(f"figdl-{key}"):
            for ext in ("png", "pdf"):
                p = rs.find(f"{item.stem}.{ext}")
                if p: st.download_button(f"Download {ext.upper()}", open(p, "rb").read(), file_name=os.path.basename(p), mime=mime(p),
                                         key=f"dl_{key}_{item.stem}_{ext}")


def gallery(exp, rs: results.ResultSet, key: str, source_id: str) -> None:
    items = figure_items(exp, rs)
    labels = [X.CATEGORY_LABEL[c] for c in X.CATEGORY_LABEL if any(i.category == c for i in items)]
    ck, qk = f"cats_{key}", f"q_{key}"
    if not isinstance(ss.get(ck), list): ss[ck] = list(labels)
    ss[ck] = [l for l in ss[ck] if l in labels]
    c1, c2 = st.columns([3, 1.4], vertical_alignment="center")
    with c1: picked = st.segmented_control("Show", labels, selection_mode="multi", key=ck, label_visibility="collapsed") or []
    with c2: query = st.text_input("Search figures", key=qk, placeholder="Search figures", label_visibility="collapsed")
    q = (query or "").strip().lower()
    shown = [i for i in items if X.CATEGORY_LABEL[i.category] in picked and (not q or q in i.title.lower() or q in i.stem.lower())]
    counts = Counter(i.category for i in items)
    md('<div class="k-group-chips">' + "".join(f'<span>{X.CATEGORY_LABEL[c]}<b>{counts[c]}</b></span>' for c in X.CATEGORY_LABEL if counts.get(c))
       + "</div>")
    md(f'<span class="k-count">{len(items)} figures in these results ({results.original_figure_files(rs)} files, PNG and PDF), '
       f'{len(shown)} shown.</span>')
    current = next((i for i in items if i.stem == ss.fig), None)
    if current: figure_detail(current, shown if current in shown else items, rs, key, source_id)
    if not shown:
        st.info("No figures match. Choose more groups or clear the search." if items else "There are no figures in these results yet."); return
    cols = st.columns(3)
    for n, it in enumerate(shown):
        with cols[n % 3], st.container(key=f"kq-card-{key}-{n}"):
            img = rs.thumb(it.stem)
            if img: show_image(img)
            else: md('<div class="k-muted" style="height:178px;display:flex;align-items:center;justify-content:center">Preview not available</div>')
            md(f'<div class="k-title">{it.title}</div><div class="k-row">{theme.badge(it.category, X.CATEGORY_LABEL[it.category])}</div>')
            with row(f"card-{key}-{n}"):
                st.button("Open", key=f"open_{key}_{it.stem}", on_click=nav, kwargs=dict(fig=it.stem))
                st.button("Compare", key=f"cmpb_{key}_{it.stem}", on_click=pin, args=("L", source_id, it.stem))
    missing = missing_items(exp, rs)
    if missing:
        with st.expander(f"{len(missing)} figure{'s' if len(missing) > 1 else ''} not in these results yet"):
            for f in missing:
                name = next((a.name for a in R.ANALYSES if a.id == f.analysis), f.analysis)
                md(f'<div class="k-sub">{f.title}: run "{name}" to make it.</div>')


TABLE_GROUP_ORDER = ("Results", "Your run", "Validation", "Comparison", "Deprecated models", "Earlier versions")


def table_note(name: str, group: str) -> str:
    base = ARCH_RE.match(name).group(2).split("__")[-1] if ARCH_RE.match(name) else name
    note = X.TABLE_NOTES.get(base, "")
    if ARCH_RE.match(name):
        return (note + " " if note else "") + f"From {X.ARCHIVE_LABEL.get(name.split('__')[0], 'an earlier version')}, kept for the record."
    if group == "Validation": return note or "A validation output of the Experiment 2 pipeline (see the verification report under Documents)."
    if group == "Deprecated models": return note or "An output of a deprecated model, kept so it can still be inspected."
    return note


def tables_view(key: str, rs: results.ResultSet) -> None:
    names = rs.tables()
    if not names: st.info("There are no tables in these results yet."); return
    by_group = {}
    for n in names: by_group.setdefault(results.group_of(rs, n), []).append(n)
    groups = [g for g in TABLE_GROUP_ORDER if g in by_group] + [g for g in by_group if g not in TABLE_GROUP_ORDER]
    gk = f"tgrp_{key}"
    if ss.get(gk) not in groups: ss[gk] = groups[0]
    if ss.get(gk + "_last") not in groups: ss[gk + "_last"] = ss[gk]
    with st.container(key=f"kp-tab-{key}"):
        md(f'<h3>Tables</h3><div class="k-group-chips">' + "".join(f"<span>{g}<b>{len(by_group[g])}</b></span>" for g in groups)
           + f'</div><span class="k-count">{len(names)} tables in these results.</span>')
        if len(groups) > 1:
            st.segmented_control("Group", groups, key=gk, label_visibility="collapsed", on_change=keep, args=(gk, gk + "_last"))
        group = ss.get(gk) or ss[gk + "_last"]
        labels = {(archive_label(n) if ARCH_RE.match(n) else n): n for n in by_group[group]}
        tk = f"tab_{key}"
        if ss.get(tk) not in labels: ss[tk] = next(iter(labels))
        label = st.selectbox("Table", list(labels), key=tk)
        name = labels[label]; note = table_note(name, group)
        if note: md(f'<p class="k-sub">{note}</p>')
        df = rs.table(name)
        if df is None: st.warning("This table could not be read."); return
        md(f'<span class="k-count">{len(df):,} rows, {df.shape[1]} columns.</span>')
        show_df(df)
        st.download_button("Download CSV", open(rs.find(name), "rb").read(), file_name=name, mime="text/csv", key=f"dlt_{key}_{name}")


def doc_label(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    m = ARCH_RE.match(stem)
    core = (m.group(2).split("__")[-1] if m else stem).replace("_", " ")
    pretty = core[:1].upper() + core[1:]
    return f"{pretty} ({X.ARCHIVE_LABEL.get(m.group(1), m.group(1))})" if m else pretty


def documents_view(key: str) -> None:
    docs = results.documents(key)
    if not docs: st.info("No documents for this view."); return
    groups = list(dict.fromkeys(g for g, _, _ in docs))
    gk = f"dgrp_{key}"
    if ss.get(gk) not in groups: ss[gk] = groups[0]
    if ss.get(gk + "_last") not in groups: ss[gk + "_last"] = ss[gk]
    with st.container(key=f"kp-docs-{key}"):
        md('<h3>Documents</h3><div class="k-group-chips">' + "".join(
            f"<span>{g}<b>{sum(1 for x in docs if x[0] == g)}</b></span>" for g in groups) + "</div>"
           '<p class="k-sub">The reports, guides and records kept in the repository, readable here and downloadable.</p>')
        if len(groups) > 1:
            st.segmented_control("Group", groups, key=gk, label_visibility="collapsed", on_change=keep, args=(gk, gk + "_last"))
        group = ss.get(gk) or ss[gk + "_last"]
        labels = {}
        for g, n, p in docs:
            if g == group: labels[doc_label(n)] = (n, p)
        dk = f"doc_{key}"
        if ss.get(dk) not in labels: ss[dk] = next(iter(labels))
        label = st.selectbox("Document", list(labels), key=dk)
    name, path = labels[label]
    with st.container(key=f"kp-doc-{key}"):
        if name.endswith(".md"):
            try:
                st.markdown(open(path, encoding="utf-8", errors="replace").read())
            except OSError as exc:
                st.warning(f"The document could not be read: {exc}")
        else:
            prev = results.pdf_preview(path)
            pages = results.pdf_pages(path)
            md(f'<span class="k-count">PDF, {pages} page{"s" if pages != 1 else ""}. The first page is shown; download it to read the rest.</span>'
               if pages else '<span class="k-count">PDF document.</span>')
            if prev: show_image(prev)
        st.download_button("Download", open(path, "rb").read(), file_name=name, mime=mime(name) if name.endswith(".pdf") else "text/markdown",
                           key=f"dld_{key}_{name}")


def side_by_side() -> None:
    with st.container(key="kp-sbs-presets"):
        md('<h3>Side by side</h3><p class="k-sub">Put any two figures next to each other: the same figure for both experiments, or two '
           "different figures. Start from a pair below, or press Compare on any figure.</p>")
        cols = st.columns(4)
        for n, (label, stem) in enumerate(X.SIDE_BY_SIDE_PRESETS):
            cols[n % 4].button(label, key=f"preset_{stem}", on_click=preset, args=(stem,))
    opts = source_options()
    panes = st.columns(2, gap="medium"); chosen = {}
    defaults = {"L": ("E1:ref", "NDT_barchart_bayesian"), "R": ("E2:ref", "NDT_barchart_bayesian")}
    for side, pane in zip(("L", "R"), panes):
        sk, fk = f"sbs_src_{side}", f"sbs_fig_{side}"
        src0, stem0 = ss.sbs.get(side) or defaults[side]
        labels = list(opts.values()); label_to_id = {v: k for k, v in opts.items()}
        if ss.get(sk) not in labels: ss[sk] = opts.get(src0, opts[defaults[side][0]])
        source_id = label_to_id[ss[sk]]
        exp, rs = source_by_id(source_id)
        items = figure_items(exp, rs); by_title = {i.title: i for i in items}
        if ss.get(fk) not in by_title:
            ss[fk] = next((i.title for i in items if i.stem == stem0), items[0].title if items else None)
        with pane, st.container(key=f"kp-pane-{side}"):
            st.selectbox("Results", labels, key=sk, label_visibility="collapsed")
            if not items: st.info("No figures in these results."); continue
            st.selectbox("Figure", list(by_title), key=fk, label_visibility="collapsed")
            it = by_title[ss[fk]]; ss.sbs[side] = (source_id, it.stem); chosen[side] = (exp, rs, it.stem, ss[sk])
            img = rs.thumb(it.stem, 1100)
            if img: show_image(img)
            else: st.info("A preview is not available; download the PDF from the figure's page.")
            md(f'<div class="k-row"><span class="k-pane-title">{it.title}</span>{theme.badge(it.category, X.CATEGORY_LABEL[it.category])}</div>')
            if it.fig: md(f'<p class="k-sub" style="margin-top:6px">{it.fig.what}</p><p class="k-muted">{it.fig.meaning}</p>')
    if "L" in chosen and "R" in chosen:
        heading, notes = I.compare(chosen["L"], chosen["R"])
        with st.container(key="kp-sbs-notes"):
            md(f"<h3>{heading}</h3><ul class='k-facts'>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>")


def sync_query(want: dict) -> None:
    """Keep the address bar in step with the page for bookmarks, writing only what changed (an unconditional write
    makes some Streamlit versions rerun the script on every write, which never stops)."""
    try:
        current = {k: st.query_params.get(k) for k in ("exp", "view", "cview")}
        if any(current.get(k) != v for k, v in want.items()) or any(current.get(k) for k in ("exp", "view", "cview") if k not in want):
            st.query_params.clear(); st.query_params.update(want)
    except Exception:
        pass


# --------------------------------------------------------------------------- page
header()
top = ss.exp_w or ss.exp_last
if top == "Compare":
    E1, E2 = R.EXPERIMENTS["E1"], R.EXPERIMENTS["E2"]
    theme.hero("Experiment 1 and Experiment 2", "#2f6bff", "Side by side",
               "The same analyses on both cohorts. Experiment 1 had a stationary target and Experiment 2 has only moving ones, so "
               "the speeds they share are 75 and 150 deg/s. On the rings, Experiment 1's targets move on the inner ring and "
               "Experiment 2's on the outer ring.",
               [("Inner ring, Experiment 1", "0 / 75 / 150 deg/s"), ("Outer ring, Experiment 2", "75 / 100 / 125 / 150 deg/s")],
               (75, 150), R.SPEED_COLOURS,
               orbit=theme.orbit_compare_svg(E1.speeds, E2.speeds, R.SPEED_COLOURS, E1.accent, E2.accent))
    st.write("")
    st.segmented_control("View", CVIEWS, key="cview_w", label_visibility="collapsed", on_change=keep, args=("cview_w", "cview_last"))
    cview = ss.cview_w or ss.cview_last
    if cview == "Side by side": guarded(side_by_side)
    elif cview == "Figures": guarded(gallery, None, results.compare_reference(), "CMP", "CMP:ref")
    elif cview == "Tables": guarded(tables_view, "CMP", results.compare_reference())
    else: guarded(documents_view, "CMP")
    sync_query({"exp": "compare", "cview": cview})
else:
    exp = R.EXPERIMENTS[EXP_IDS[top]]
    guarded(hero_for, exp, result_set(exp))
    st.write("")
    st.segmented_control("Section", VIEWS, key="view_w", label_visibility="collapsed", on_change=keep, args=("view_w", "view_last"))
    view = ss.view_w or ss.view_last
    if view == "Overview": guarded(overview, exp)
    elif view == "Run": guarded(run_view, exp)
    elif view == "Figures":
        source_switch(exp); rs = result_set(exp)
        guarded(gallery, exp, rs, exp.id, f"{exp.id}:{'run' if rs.source == 'run' else 'ref'}")
    elif view == "Tables":
        source_switch(exp); guarded(tables_view, exp.id, result_set(exp))
    elif view == "Documents": guarded(documents_view, exp.id)
    elif view == "Models": guarded(models_view, exp)
    sync_query({"exp": exp.id, "view": view})

ok, bad = sync_status()
md('<p class="k-muted" style="margin-top:28px">' + (f"{ok} pipeline scripts, byte-identical to the repository versions recorded in MANIFEST.json."
                                                     if bad == 0 else f"{bad} pipeline script(s) differ from MANIFEST.json; run tools/sync_pipelines.py.")
   + f" Runs are stored in {runner.runs_root()}. Running on {env_summary()}.</p>")
