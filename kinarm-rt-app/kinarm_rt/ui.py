"""
UI design system for the app: a modern, cohesive look with smooth animations.

Everything here is presentation only. It injects a stylesheet and provides a few
HTML helpers (hero header, animated step indicator, section headers, metric
cards) so the app reads as a polished product rather than a default Streamlit page.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens + component styles + animations
# --------------------------------------------------------------------------- #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --kx-bg:#f5f6fb; --kx-surface:#ffffff; --kx-ink:#1e2233; --kx-muted:#6b7185;
  --kx-line:#e7e9f2; --kx-primary:#5b5bd6; --kx-primary-2:#7c6cf0; --kx-primary-ink:#4b46c9;
  --kx-green:#5aa469; --kx-amber:#c98a2b; --kx-red:#c0504d;
  --kx-shadow:0 1px 2px rgba(20,24,45,.04),0 8px 24px rgba(20,24,45,.06);
  --kx-shadow-lg:0 2px 6px rgba(20,24,45,.05),0 18px 48px rgba(20,24,45,.10);
  --kx-radius:16px;
}

html, body, [class*="css"], .stMarkdown, .stApp{
  font-family:'Inter',ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{ background:
  radial-gradient(1200px 600px at 15% -10%, #eef0ff 0%, rgba(238,240,255,0) 55%),
  radial-gradient(1000px 500px at 100% 0%, #f2ecff 0%, rgba(242,236,255,0) 50%),
  var(--kx-bg); }

/* roomier main column */
.block-container{ max-width:1120px; padding-top:2.2rem; padding-bottom:4rem; }

/* keep Streamlit's top chrome unobtrusive */
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }

/* ---- entrance animations ---- */
@keyframes kxFadeUp{ from{ opacity:0; transform:translateY(10px);} to{ opacity:1; transform:none;} }
@keyframes kxFade{ from{opacity:0;} to{opacity:1;} }
@keyframes kxShine{ 0%{background-position:-200% 0;} 100%{background-position:200% 0;} }
@keyframes kxPop{ 0%{transform:scale(.6);opacity:0;} 60%{transform:scale(1.08);} 100%{transform:scale(1);opacity:1;} }
.block-container > div{ animation:kxFadeUp .5s cubic-bezier(.22,.61,.36,1) both; }

/* ---- hero ---- */
.kx-hero{
  position:relative; overflow:hidden; border-radius:22px; padding:30px 34px; margin-bottom:22px;
  color:#fff; background:linear-gradient(120deg,#5b5bd6 0%, #7c6cf0 45%, #9b6cf0 100%);
  background-size:180% 180%; animation:kxShine 14s ease infinite, kxFadeUp .6s ease both;
  box-shadow:var(--kx-shadow-lg);
}
.kx-hero:before{ content:""; position:absolute; inset:0;
  background:radial-gradient(600px 220px at 90% -30%, rgba(255,255,255,.35), transparent 60%); }
.kx-hero h1{ margin:0; font-size:30px; font-weight:800; letter-spacing:-.02em; }
.kx-hero p{ margin:.5rem 0 0; font-size:15px; line-height:1.5; color:rgba(255,255,255,.92); max-width:760px; }
.kx-badges{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.kx-badge{ font-size:12px; font-weight:600; padding:5px 11px; border-radius:999px;
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28); backdrop-filter:blur(4px); }

/* ---- stepper ---- */
.kx-steps{ display:flex; align-items:flex-start; gap:0; margin:6px 0 20px; }
.kx-step{ flex:1; text-align:center; position:relative; }
.kx-step .dot{ width:34px; height:34px; border-radius:50%; margin:0 auto 8px; display:flex;
  align-items:center; justify-content:center; font-weight:700; font-size:14px;
  background:#fff; color:var(--kx-muted); border:2px solid var(--kx-line);
  transition:all .35s cubic-bezier(.22,.61,.36,1); box-shadow:var(--kx-shadow); }
.kx-step .lbl{ font-size:12.5px; font-weight:600; color:var(--kx-muted); transition:color .3s; }
.kx-step:before{ content:""; position:absolute; top:17px; left:-50%; width:100%; height:2px;
  background:var(--kx-line); z-index:0; transition:background .4s; }
.kx-step:first-child:before{ display:none; }
.kx-step.done .dot{ background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2));
  color:#fff; border-color:transparent; animation:kxPop .4s ease both; }
.kx-step.done .lbl{ color:var(--kx-ink); }
.kx-step.done:before{ background:linear-gradient(90deg,var(--kx-primary),var(--kx-primary-2)); }
.kx-step.active .dot{ border-color:var(--kx-primary); color:var(--kx-primary-ink);
  box-shadow:0 0 0 5px rgba(91,91,214,.14); }
.kx-step.active .lbl{ color:var(--kx-ink); }

/* ---- section headers ---- */
.kx-sec{ display:flex; align-items:center; gap:12px; margin:6px 0 12px; }
.kx-sec .ico{ width:38px; height:38px; border-radius:11px; display:flex; align-items:center;
  justify-content:center; font-size:19px; color:#fff;
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2)); box-shadow:var(--kx-shadow); }
.kx-sec .t{ font-size:19px; font-weight:700; color:var(--kx-ink); letter-spacing:-.01em; }
.kx-sec .d{ font-size:13px; color:var(--kx-muted); margin-top:1px; }

/* ---- cards: style bordered containers ---- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--kx-surface); border-radius:var(--kx-radius)!important;
  border:1px solid var(--kx-line)!important; box-shadow:var(--kx-shadow);
  padding:6px 4px; transition:box-shadow .25s ease, transform .25s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover{ box-shadow:var(--kx-shadow-lg); }

/* ---- buttons ---- */
.stButton>button, .stDownloadButton>button{
  border-radius:11px; font-weight:600; border:1px solid var(--kx-line);
  padding:.5rem 1rem; transition:transform .15s ease, box-shadow .2s ease, background .2s ease;
  background:#fff; color:var(--kx-ink); }
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-1px); box-shadow:var(--kx-shadow-lg); border-color:#d9dcec; }
.stButton>button:active, .stDownloadButton>button:active{ transform:translateY(0); }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2));
  color:#fff; border:none; box-shadow:0 6px 16px rgba(91,91,214,.30); }
.stButton>button[kind="primary"]:hover{ box-shadow:0 10px 26px rgba(91,91,214,.42); }

/* ---- metrics as cards ---- */
[data-testid="stMetric"]{ background:var(--kx-surface); border:1px solid var(--kx-line);
  border-radius:14px; padding:14px 16px; box-shadow:var(--kx-shadow);
  transition:transform .2s ease, box-shadow .2s ease; }
[data-testid="stMetric"]:hover{ transform:translateY(-2px); box-shadow:var(--kx-shadow-lg); }
[data-testid="stMetricValue"]{ font-weight:700; font-size:26px; color:var(--kx-primary-ink); }
[data-testid="stMetricLabel"]{ color:var(--kx-muted); font-weight:600; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--kx-line); }
.stTabs [data-baseweb="tab"]{ height:42px; border-radius:10px 10px 0 0; padding:0 16px;
  font-weight:600; color:var(--kx-muted); transition:all .2s ease; }
.stTabs [data-baseweb="tab"]:hover{ color:var(--kx-ink); background:rgba(91,91,214,.06); }
.stTabs [aria-selected="true"]{ color:var(--kx-primary-ink)!important; background:rgba(91,91,214,.08); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--kx-primary); height:3px; border-radius:3px; }

/* ---- dataframes ---- */
[data-testid="stDataFrame"], [data-testid="stTable"]{ border-radius:12px; overflow:hidden;
  border:1px solid var(--kx-line); box-shadow:var(--kx-shadow); }

/* ---- expanders ---- */
[data-testid="stExpander"]{ border:1px solid var(--kx-line)!important; border-radius:12px!important;
  box-shadow:var(--kx-shadow); background:var(--kx-surface); overflow:hidden; }
[data-testid="stExpander"] summary:hover{ color:var(--kx-primary-ink); }

/* ---- alerts / status ---- */
.stAlert{ border-radius:12px; border:1px solid var(--kx-line); box-shadow:var(--kx-shadow); }
[data-testid="stStatusWidget"], [data-testid="stStatus"]{ border-radius:12px!important; }

/* ---- inputs ---- */
[data-testid="stFileUploaderDropzone"]{ border-radius:14px; border:1.5px dashed #c7cbe4;
  background:linear-gradient(180deg,#fbfbff,#f4f5fd); transition:border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--kx-primary); background:#f2f2fe; }
[data-baseweb="select"]>div, .stNumberInput input, .stTextInput input{ border-radius:10px!important; }
.stRadio [role="radiogroup"]{ gap:6px; }

/* spinner colour */
.stSpinner>div{ border-top-color:var(--kx-primary)!important; }

/* pills used in prose */
.kx-pill{ display:inline-block; font-size:12px; font-weight:600; padding:2px 9px; border-radius:999px;
  background:rgba(91,91,214,.10); color:var(--kx-primary-ink); }
.kx-note{ background:#fbfbff; border:1px solid var(--kx-line); border-left:3px solid var(--kx-primary);
  border-radius:10px; padding:11px 14px; font-size:13.5px; color:#41465c; }
</style>
"""


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str] | None = None):
    b = ""
    if badges:
        chips = "".join(f"<span class='kx-badge'>{x}</span>" for x in badges)
        b = f"<div class='kx-badges'>{chips}</div>"
    st.markdown(f"<div class='kx-hero'><h1>{title}</h1><p>{subtitle}</p>{b}</div>",
                unsafe_allow_html=True)


def stepper(labels: list[str], current: int):
    """current is the 0-based index of the active step; earlier steps show as done."""
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


def note(text: str):
    st.markdown(f"<div class='kx-note'>{text}</div>", unsafe_allow_html=True)
