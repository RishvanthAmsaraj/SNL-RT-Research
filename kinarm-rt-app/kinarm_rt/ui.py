"""
UI design system: a modern, premium look with dual light/dark themes.

Presentation only. `inject_theme(theme_type)` sets CSS variables to match the
active Streamlit theme (bright white in light mode, OLED black in dark mode, blue
accents in both) and styles every surface consistently. Helpers render the hero,
the step indicator, section headers, and callouts.
"""

from __future__ import annotations

import streamlit as st

# Per-theme design tokens.
_LIGHT = {
    "bg": "#ffffff", "surface": "#ffffff", "inset": "#f2f6ff", "ink": "#0f1729",
    "muted": "#5b6478", "line": "#e3e9f5", "line2": "#eef2fb",
    "primary": "#2f6bff", "primary2": "#5b8cff", "primaryInk": "#1f57e6",
    "glow": "rgba(47,107,255,.28)", "glowSoft": "rgba(47,107,255,.14)",
    "shadow": "0 1px 2px rgba(15,23,41,.05),0 10px 28px rgba(15,23,41,.07)",
    "shadowLg": "0 2px 8px rgba(15,23,41,.06),0 22px 55px rgba(15,23,41,.12)",
}
_DARK = {
    "bg": "#000000", "surface": "#0b0e16", "inset": "#10151f", "ink": "#e7ecf5",
    "muted": "#9aa3b8", "line": "#1c2230", "line2": "#161b26",
    "primary": "#4d8bff", "primary2": "#7aa8ff", "primaryInk": "#a9c6ff",
    "glow": "rgba(77,139,255,.40)", "glowSoft": "rgba(77,139,255,.20)",
    "shadow": "0 1px 2px rgba(0,0,0,.5),0 10px 28px rgba(0,0,0,.55)",
    "shadowLg": "0 2px 10px rgba(0,0,0,.55),0 24px 60px rgba(0,0,0,.65)",
}


def _root_vars(t: dict) -> str:
    return ";".join(f"--kx-{k}:{v}" for k, v in {
        "bg": t["bg"], "surface": t["surface"], "inset": t["inset"], "ink": t["ink"],
        "muted": t["muted"], "line": t["line"], "line2": t["line2"],
        "primary": t["primary"], "primary-2": t["primary2"], "primary-ink": t["primaryInk"],
        "glow": t["glow"], "glow-soft": t["glowSoft"], "shadow": t["shadow"], "shadow-lg": t["shadowLg"],
    }.items())


_STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stApp, input, button, textarea, select{
  font-family:'Inter',ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{ background:var(--kx-bg); }
.block-container{ max-width:1120px; padding-top:2rem; padding-bottom:4rem; }

/* consistent spacing rhythm everywhere */
[data-testid="stVerticalBlock"]{ gap:0.8rem; }
[data-testid="stHorizontalBlock"]{ gap:1rem; }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]{ gap:0.7rem; }
hr{ margin:0.6rem 0 !important; opacity:.6; }
h4{ margin:0.2rem 0 0.4rem !important; font-weight:700; letter-spacing:-.01em; }

/* keep the ☰ menu (theme switch) but remove deploy button, running indicator, top bar */
[data-testid="stToolbar"] [data-testid="stToolbarActions"] a,
[data-testid="stAppDeployButton"], .stAppDeployButton,
.stDeployButton, [data-testid="stStatusWidget"], [data-testid="stDecoration"]{ display:none !important; }
footer{ visibility:hidden; }

/* smooth, consistent motion */
*{ transition-timing-function:cubic-bezier(.22,.61,.36,1); }
@keyframes kxFadeUp{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;} }
.block-container > div{ animation:kxFadeUp .45s both; }

/* ---------- hero (blue, glow, no shimmer) ---------- */
.kx-hero{ position:relative; overflow:hidden; border-radius:20px; padding:28px 32px; margin-bottom:18px;
  color:#fff; background:linear-gradient(120deg,var(--kx-primary),var(--kx-primary-2));
  box-shadow:0 10px 34px var(--kx-glow); animation:kxFadeUp .5s both; }
.kx-hero:before{ content:""; position:absolute; inset:0;
  background:radial-gradient(560px 200px at 88% -40%, rgba(255,255,255,.28), transparent 60%); }
.kx-hero h1{ margin:0; font-size:29px; font-weight:800; letter-spacing:-.02em; }
.kx-hero p{ margin:.5rem 0 0; font-size:14.5px; line-height:1.55; color:rgba(255,255,255,.92); max-width:770px; }
.kx-badges{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.kx-badge{ font-size:12px; font-weight:600; padding:5px 11px; border-radius:999px;
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.26); }

/* ---------- stepper (circles ON TOP of the line) ---------- */
.kx-steps{ display:flex; margin:8px 0 22px; }
.kx-step{ flex:1; text-align:center; position:relative; }
.kx-step .dot{ position:relative; z-index:2; width:34px; height:34px; border-radius:50%;
  margin:0 auto 8px; display:flex; align-items:center; justify-content:center; font-weight:700;
  font-size:14px; background:var(--kx-surface); color:var(--kx-muted);
  border:2px solid var(--kx-line); transition:all .35s; box-shadow:var(--kx-shadow); }
.kx-step .lbl{ font-size:12.5px; font-weight:600; color:var(--kx-muted); transition:color .3s; }
.kx-step:not(:first-child):before{ content:""; position:absolute; z-index:0; top:17px; left:-50%;
  width:100%; height:2px; background:var(--kx-line); transition:background .4s; }
.kx-step.done .dot{ background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2));
  color:#fff; border-color:transparent; box-shadow:0 4px 14px var(--kx-glow); }
.kx-step.done .lbl{ color:var(--kx-ink); }
.kx-step.done:before{ background:linear-gradient(90deg,var(--kx-primary),var(--kx-primary-2)); }
.kx-step.active .dot{ border-color:var(--kx-primary); color:var(--kx-primary-ink);
  box-shadow:0 0 0 5px var(--kx-glow-soft); }
.kx-step.active .lbl{ color:var(--kx-ink); }

/* ---------- section headers ---------- */
.kx-sec{ display:flex; align-items:center; gap:12px; margin:2px 0 14px; }
.kx-sec .ico{ width:36px; height:36px; border-radius:10px; display:flex; align-items:center;
  justify-content:center; font-size:16px; font-weight:700; color:#fff;
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2));
  box-shadow:0 4px 12px var(--kx-glow); }
.kx-sec .t{ font-size:18px; font-weight:700; color:var(--kx-ink); letter-spacing:-.01em; }
.kx-sec .d{ font-size:12.5px; color:var(--kx-muted); margin-top:1px; }

/* sub-labels inside a card */
.kx-eyebrow{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--kx-primary-ink); margin:6px 0 8px; }

/* ---------- cards (bordered containers), consistent padding ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--kx-surface);
  border-radius:16px !important; border:1px solid var(--kx-line) !important;
  box-shadow:var(--kx-shadow); padding:20px 22px !important; margin-bottom:6px;
  transition:box-shadow .25s, transform .25s; }
[data-testid="stVerticalBlockBorderWrapper"]:hover{ box-shadow:var(--kx-shadow-lg); }

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button{
  border-radius:11px; font-weight:600; border:1px solid var(--kx-line); padding:.5rem 1rem;
  background:var(--kx-surface); color:var(--kx-ink);
  transition:transform .15s, box-shadow .2s, background .2s, border-color .2s; }
.stButton>button:hover, .stDownloadButton>button:hover{ transform:translateY(-1px);
  box-shadow:var(--kx-shadow-lg); border-color:var(--kx-primary); }
.stButton>button:active, .stDownloadButton>button:active{ transform:translateY(0); }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2)); color:#fff; border:none;
  box-shadow:0 6px 18px var(--kx-glow); }
.stButton>button[kind="primary"]:hover{ box-shadow:0 10px 28px var(--kx-glow); }

/* ---------- segmented control (button toggles) ---------- */
[data-testid="stSegmentedControl"]{ margin-top:2px; }
[data-testid="stSegmentedControl"] button{ border-radius:10px !important; font-weight:600;
  border:1px solid var(--kx-line); background:var(--kx-surface); color:var(--kx-muted);
  transition:all .18s; }
[data-testid="stSegmentedControl"] button:hover{ color:var(--kx-ink); border-color:var(--kx-primary); }
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmentedControlActive"]{
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2)) !important;
  color:#fff !important; border-color:transparent !important; box-shadow:0 4px 12px var(--kx-glow); }

/* ---------- metrics as cards ---------- */
[data-testid="stMetric"]{ background:var(--kx-inset); border:1px solid var(--kx-line);
  border-radius:14px; padding:14px 16px; box-shadow:var(--kx-shadow); transition:transform .2s, box-shadow .2s; }
[data-testid="stMetric"]:hover{ transform:translateY(-2px); box-shadow:var(--kx-shadow-lg); }
[data-testid="stMetricValue"]{ font-weight:700; font-size:25px; color:var(--kx-primary-ink); }
[data-testid="stMetricLabel"]{ color:var(--kx-muted); font-weight:600; }

/* ---------- tabs (rounded pills, consistent with segmented control) ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:none; padding-bottom:4px; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"]{ height:38px; border-radius:10px; padding:0 15px; font-weight:600;
  color:var(--kx-muted); background:transparent; transition:all .2s; }
.stTabs [data-baseweb="tab"]:hover{ color:var(--kx-ink); background:var(--kx-glow-soft); }
.stTabs [aria-selected="true"]{ color:#fff !important;
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2)) !important;
  box-shadow:0 4px 12px var(--kx-glow); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none !important; }

/* ---------- dataframes / tables ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"]{ border-radius:12px; overflow:hidden;
  border:1px solid var(--kx-line); box-shadow:var(--kx-shadow); }

/* ---------- expanders ---------- */
[data-testid="stExpander"]{ border:1px solid var(--kx-line) !important; border-radius:12px !important;
  box-shadow:var(--kx-shadow); background:var(--kx-surface); overflow:hidden; }
[data-testid="stExpander"] summary:hover{ color:var(--kx-primary-ink); }

/* ---------- alerts / status ---------- */
.stAlert{ border-radius:12px; border:1px solid var(--kx-line); box-shadow:var(--kx-shadow); }

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"]{ border-radius:14px; border:1.5px dashed var(--kx-line);
  background:var(--kx-inset); transition:border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--kx-primary); }
[data-baseweb="select"]>div, .stNumberInput input, .stTextInput input, .stTextInput>div>div{
  border-radius:10px !important; }
.stSlider [data-baseweb="slider"] [role="slider"]{ box-shadow:0 2px 8px var(--kx-glow); }
.stSpinner>div{ border-top-color:var(--kx-primary) !important; }

/* ---------- helper chips / notes ---------- */
.kx-pill{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; padding:3px 10px; border-radius:999px;
  background:var(--kx-glow-soft); color:var(--kx-primary-ink); }
.kx-note{ background:var(--kx-inset); border:1px solid var(--kx-line); border-left:3px solid var(--kx-primary);
  border-radius:10px; padding:11px 14px; font-size:13px; color:var(--kx-ink); line-height:1.5; }
.kx-hint{ font-size:12px; color:var(--kx-muted); margin:-2px 0 8px; line-height:1.45; }
"""


def current_theme() -> str:
    try:
        t = getattr(st.context, "theme", None)
        typ = getattr(t, "type", None)
        return "dark" if typ == "dark" else "light"
    except Exception:
        return "light"


def inject_theme(theme_type: str | None = None):
    theme_type = theme_type or current_theme()
    tok = _DARK if theme_type == "dark" else _LIGHT
    st.markdown(f"<style>:root{{{_root_vars(tok)}}}\n{_STATIC_CSS}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str] | None = None):
    b = ""
    if badges:
        chips = "".join(f"<span class='kx-badge'>{x}</span>" for x in badges)
        b = f"<div class='kx-badges'>{chips}</div>"
    st.markdown(f"<div class='kx-hero'><h1>{title}</h1><p>{subtitle}</p>{b}</div>",
                unsafe_allow_html=True)


def stepper(labels: list[str], current: int):
    cells = []
    for i, lab in enumerate(labels):
        state = "done" if i < current else ("active" if i == current else "")
        mark = "✓" if i < current else str(i + 1)
        cells.append(f"<div class='kx-step {state}'><div class='dot'>{mark}</div>"
                     f"<div class='lbl'>{lab}</div></div>")
    st.markdown(f"<div class='kx-steps'>{''.join(cells)}</div>", unsafe_allow_html=True)


def section(title: str, desc: str = "", icon: str = "•"):
    d = f"<div class='d'>{desc}</div>" if desc else ""
    st.markdown(f"<div class='kx-sec'><div class='ico'>{icon}</div>"
                f"<div><div class='t'>{title}</div>{d}</div></div>", unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f"<div class='kx-eyebrow'>{text}</div>", unsafe_allow_html=True)


def note(text: str):
    st.markdown(f"<div class='kx-note'>{text}</div>", unsafe_allow_html=True)


def hint(text: str):
    st.markdown(f"<div class='kx-hint'>{text}</div>", unsafe_allow_html=True)
