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
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"],
[data-testid="stLayoutWrapper"]:has(.kx-sec) > [data-testid="stVerticalBlock"]{ gap:0.7rem; }
hr{ margin:0.6rem 0 !important; opacity:.6; }
h4{ margin:0.2rem 0 0.4rem !important; font-weight:700; letter-spacing:-.01em; }

/* keep the ☰ menu (theme switch) but remove deploy button, running indicator, top bar */
[data-testid="stToolbar"] [data-testid="stToolbarActions"] a,
[data-testid="stAppDeployButton"], .stAppDeployButton,
.stDeployButton, [data-testid="stStatusWidget"], [data-testid="stDecoration"]{ display:none !important; }
footer{ visibility:hidden; }

/* smooth, consistent motion */
*{ transition-timing-function:cubic-bezier(.22,.61,.36,1); }
@keyframes kxFadeUp{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;} }
@keyframes kxFade{ from{opacity:0;} to{opacity:1;} }
@keyframes kxRise{ from{opacity:0; transform:translateY(6px) scale(.995);} to{opacity:1; transform:none;} }
/* Section cards (steps 1-4) rise in with a stagger, so moving from one step to the
   next feels like it arrives rather than blinking on.

   Streamlit renames the container wrapper between versions: older builds emit
   [data-testid="stVerticalBlockBorderWrapper"], current builds emit
   [data-testid="stLayoutWrapper"] -- but that one wraps EVERY layout block
   (columns included), so it is narrowed with :has(.kx-sec) to the four step cards,
   each of which renders a section header. Both are listed so the app looks and
   behaves the same whichever version is installed.

   Fill mode is `backwards`, not `both`: the transform applies during the delay and
   the animation, then the element returns to its normal untransformed state. That
   matters because ANY retained transform on an ancestor of a figure establishes a
   containing block, which collapses Streamlit's position:fixed fullscreen overlay. */
.block-container [data-testid="stVerticalBlockBorderWrapper"],
.block-container [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec){
  animation:kxSection .62s cubic-bezier(.16,.84,.34,1) backwards; }
@keyframes kxSection{
  from{ opacity:0; transform:translateY(26px); }
  60%{ opacity:1; }
  to{ opacity:1; transform:translateY(0); }
}
/* Stagger: the general sibling combinator counts how many step cards precede this
   one, so the delays hold regardless of where the cards sit among their siblings
   (hero, stepper, and -- only when PyMC is missing -- a warning come first). */
.block-container [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"],
.block-container [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec){ animation-delay:.06s; }
.block-container [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"],
.block-container [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec){ animation-delay:.12s; }
.block-container [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"] ~ [data-testid="stVerticalBlockBorderWrapper"],
.block-container [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec) ~ [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec){ animation-delay:.18s; }
/* NOTE: a blanket `.block-container > div{ animation:kxFade .5s both }` used to live
   here with staggered nth-child delays. It broke fullscreen: with fill-mode `both`
   and a delay, a matched div holds the from-state (opacity 0) during its delay, so
   it is invisible but still laid out and still takes pointer events. Streamlit
   re-indexes these divs when the fullscreen overlay mounts, which left a phantom
   transparent div sitting over the expand/minimise control -- clicks landed on it
   instead of the button, and Escape became the only way out. Section entrances are
   handled by the scoped .kx-sec rule above instead. */
/* tab panels ease in when you switch tabs inside Results */
.stTabs [data-testid="stTabPanel"]{ animation:kxFade .32s backwards; }
/* content eases in with opacity only — a lingering transform on any ancestor of a
   figure or table would collapse Streamlit's fixed fullscreen overlay */
/* Figures are deliberately NOT animated: [data-testid="stImage"] hosts the
   fullscreen frame, and any filled animation on it leaves styles applied for
   good, which puts a transparent layer over the minimise control. */
[data-testid="stDataFrame"], [data-testid="stMetric"]{ animation:kxFade .45s backwards; }
/* honour reduced-motion preferences */
@media (prefers-reduced-motion: reduce){
  *, .block-container [data-testid="stVerticalBlockBorderWrapper"],
  .block-container [data-testid="stLayoutWrapper"], .block-container > div,
  .stTabs [data-testid="stTabPanel"], .kx-hero{ animation:none !important; transition:none !important; }
}
[data-testid="stExpander"]{ transition:box-shadow .2s, border-color .2s; }
[data-testid="stExpander"]:hover{ box-shadow:var(--kx-shadow-lg); border-color:var(--kx-primary) !important; }

/* ---------- hero (blue, glow, no shimmer) ---------- */
.kx-hero{ position:relative; overflow:hidden; border-radius:20px; padding:28px 32px; margin-bottom:18px;
  color:#fff; background:linear-gradient(120deg,var(--kx-primary),var(--kx-primary-2));
  box-shadow:0 10px 34px var(--kx-glow); animation:kxFadeUp .5s backwards; }
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
  color:var(--kx-primary-ink); margin:10px 0 6px; }

/* ---------- cards (bordered containers), consistent padding ---------- */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec){ background:var(--kx-surface);
  border-radius:16px !important; border:1px solid var(--kx-line) !important;
  box-shadow:var(--kx-shadow); padding:20px 22px !important; margin-bottom:6px;
  transition:box-shadow .25s, transform .25s; }
[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] .kx-sec):hover{ box-shadow:var(--kx-shadow-lg); }

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

/* ---------- tabs (correct selectors for this Streamlit build) ---------- */
.stTabs [role="tablist"]{ gap:6px; border-bottom:none !important; padding-bottom:4px; flex-wrap:wrap; }
.stTabs [data-testid="stTab"]{ border-radius:10px !important; padding:7px 15px !important;
  font-weight:600; color:var(--kx-muted); background:transparent; border-bottom:none !important;
  transition:background .2s, color .2s; }
.stTabs [data-testid="stTab"]:hover{ color:var(--kx-ink); background:var(--kx-glow-soft); }
/* the active tab is a filled pill -- no glow behind it, the fill is the highlight */
.stTabs [data-testid="stTab"][aria-selected="true"]{ color:#fff !important;
  background:linear-gradient(135deg,var(--kx-primary),var(--kx-primary-2)) !important;
  box-shadow:none !important; border-bottom:none !important; }
.stTabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p{ font-weight:600; margin:0; }
/* Breathing room inside the results tabs.
   Section 4 holds seven tabs, each with a stack of figures, tables and captions.
   Without explicit rhythm the panels read as one dense block, so the tab strip is
   separated from its content, figures are spaced from each other, and headings get
   room above them. */
.stTabs [role="tablist"]{ gap:8px; padding:0 0 10px; margin-bottom:6px;
  border-bottom:1px solid var(--kx-line) !important; }
.stTabs [data-testid="stTabPanel"]{ padding-top:22px; padding-bottom:8px; }
.stTabs [data-testid="stTabPanel"] h4{ margin:1.1rem 0 .35rem !important; font-size:16px; }
.stTabs [data-testid="stTabPanel"] [data-testid="stCaptionContainer"]{ margin-bottom:.45rem; }
/* space between stacked figures, and between a figure and whatever follows it */
.stTabs [data-testid="stTabPanel"] [data-testid="stImage"]{ margin:10px 0 26px; }
.stTabs [data-testid="stTabPanel"] [data-testid="stImage"] + [data-testid="stImage"]{
  margin-top:4px; }
/* tables and expanders inside a tab get the same rhythm */
.stTabs [data-testid="stTabPanel"] [data-testid="stDataFrame"]{ margin-bottom:20px; }
.stTabs [data-testid="stTabPanel"] [data-testid="stExpander"]{ margin:6px 0 20px; }
.stTabs [data-testid="stTabPanel"] .kx-eyebrow{ margin-top:22px; }
.stTabs [data-testid="stTabPanel"] > div > div > [data-testid="stVerticalBlock"]{ gap:.55rem; }
/* the buttons in Advanced / Download sit closer to what they produce */
.stTabs [data-testid="stTabPanel"] [data-testid="stButton"]{ margin-bottom:10px; }
/* remove the sliding/underline indicators (the active tab is a filled pill instead) */
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before{ display:none !important; }
.stTabs .react-aria-SelectionIndicator{ display:none !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none !important; }

/* ---------- tooltip "?" sits right next to the label, consistently ---------- */
[data-testid="stWidgetLabel"]{ display:flex !important; align-items:center; gap:6px; }
[data-testid="stWidgetLabel"] > *{ flex:0 0 auto !important; margin:0 !important; }
[data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"]{ width:auto !important; }
[data-testid="stTooltipIcon"]{ display:inline-flex !important; align-items:center; }
[data-testid="stTooltipIcon"] svg{ transition:color .2s, transform .2s; }
[data-testid="stTooltipIcon"]:hover svg{ color:var(--kx-primary); transform:scale(1.12); }

/* ---------- figures rendered as images (style the img, never the container, so
   Streamlit's native fullscreen — which fills the viewport — keeps working) ---------- */
[data-testid="stImage"] img{ border-radius:10px; border:1px solid var(--kx-line);
  box-shadow:var(--kx-shadow); transition:box-shadow .25s; }
[data-testid="stImage"]:hover img{ box-shadow:var(--kx-shadow-lg); }

/* Expand / minimise control.
   Streamlit renames this button between versions: older builds use
   [data-testid="StyledFullScreenButton"], current builds use
   [data-testid="stBaseButton-elementToolbar"] with aria-label
   "Fullscreen" / "Close fullscreen". All of them are targeted so the control is
   styled whichever version is installed.

   Only its APPEARANCE is changed. Its box is left alone deliberately: forcing a
   larger width/height pushed the button out from under its own toolbar, so the
   topmost element at the button's centre became the image container and clicks
   never reached the button -- which left Escape as the only way out of
   fullscreen. The toolbar is given a stacking context instead, so the control
   always sits above the figure and stays clickable. */
[data-testid="stElementToolbar"],
[data-testid="stImage"] [data-testid="stElementToolbar"],
[data-testid="stFullScreenFrame"] [data-testid="stElementToolbar"]{
  opacity:1 !important; visibility:visible !important;
  z-index:3 !important; pointer-events:auto !important; }
/* Only the figure that is actually expanded gets lifted above everything. A page can
   hold a dozen figures, and giving every toolbar a huge z-index put the other
   toolbars on top of the fullscreen overlay, where they intercepted the click meant
   for the minimise control. The expanded frame is the one containing a
   "Close fullscreen" button. */
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]){
  z-index:2147483000 !important; }
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"])
  [data-testid="stElementToolbar"]{ z-index:2147483001 !important; }
[data-testid="StyledFullScreenButton"],
[data-testid="stBaseButton-elementToolbar"],
button[aria-label="Fullscreen"],
button[aria-label="Close fullscreen"]{
  opacity:1 !important; visibility:visible !important;
  pointer-events:auto !important;
  background:var(--kx-surface) !important;
  border:1px solid var(--kx-line) !important;
  border-radius:8px !important; color:var(--kx-ink) !important;
  box-shadow:var(--kx-shadow) !important;
  transition:background .18s, border-color .18s; }
[data-testid="StyledFullScreenButton"] svg,
[data-testid="stBaseButton-elementToolbar"] svg,
button[aria-label="Fullscreen"] svg,
button[aria-label="Close fullscreen"] svg{
  color:var(--kx-ink) !important; fill:currentColor !important; opacity:1 !important; }
[data-testid="StyledFullScreenButton"]:hover,
[data-testid="stBaseButton-elementToolbar"]:hover,
button[aria-label="Fullscreen"]:hover,
button[aria-label="Close fullscreen"]:hover{
  background:var(--kx-primary) !important; border-color:var(--kx-primary) !important; }
[data-testid="StyledFullScreenButton"]:hover svg,
[data-testid="stBaseButton-elementToolbar"]:hover svg,
button[aria-label="Fullscreen"]:hover svg,
button[aria-label="Close fullscreen"]:hover svg{ color:#fff !important; }

/* In fullscreen the control is the only thing on screen besides the figure, and
   there is nothing beneath it to intercept clicks, so it can be made properly
   large and obvious. It is deliberately NOT enlarged inline: forcing a bigger box
   there pushed the button out from under its own toolbar and clicks stopped
   landing on it, which is what left Escape as the only way out. */
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"])
  [data-testid="stElementToolbar"]{
  position:fixed !important; top:16px !important; right:20px !important; }
button[aria-label="Close fullscreen"]{
  width:44px !important; height:44px !important; padding:10px !important;
  border-radius:12px !important; border-width:1.5px !important; }
button[aria-label="Close fullscreen"] svg{ width:22px !important; height:22px !important; }

/* Inline figures fill the column and never exceed it. The frame wrapper below is
   present whether or not a figure is expanded, so any sizing rule written against
   it alone also applies inline -- which is what let a 2400px-wide PNG render at its
   natural size and overflow the section card. */
[data-testid="stImage"] img,
[data-testid="stFullScreenFrame"] img{
  max-width:100% !important; height:auto !important; display:block; }

/* Only the expanded figure is fitted to the viewport. The expanded frame is the one
   holding a "Close fullscreen" button, so this cannot leak to the inline state. */
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]) img{
  border-radius:8px; box-shadow:none; border:none;
  max-width:calc(100vw - 96px) !important; max-height:calc(100vh - 96px) !important;
  width:auto !important; height:auto !important;
  object-fit:contain; margin:0 auto; }
/* give the expanded figure a calm backdrop and room for the close control */
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]){
  background:var(--kx-bg) !important; padding:48px 24px 24px !important;
  display:flex !important; align-items:center; justify-content:center; }
/* the wrappers between the frame and the image keep their inline column width,
   which would cap the expanded figure at its in-page size -- release them so the
   figure can actually use the viewport */
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]) > div,
[data-testid="stFullScreenFrame"]:has(button[aria-label="Close fullscreen"]) [data-testid="stImage"]{
  width:100% !important; max-width:none !important;
  display:flex !important; align-items:center; justify-content:center; }

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
.kx-hint{ font-size:12.5px; color:var(--kx-muted); margin:2px 0 8px; line-height:1.5; }
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


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
def format_eta(seconds) -> str:
    """'~2m 05s left' from a number of seconds; empty string if not yet known."""
    import math
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return ""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"~{seconds}s left"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"~{m}m {s:02d}s left"
    h, m = divmod(m, 60)
    return f"~{h}h {m:02d}m left"


class StepBar:
    """
    A labelled progress bar that estimates the time remaining.

    The estimate comes from the rate actually observed so far rather than a fixed
    guess, so it adapts to the machine and to how many cells need the slower
    two-component fit. Call the instance as progress(done, total).
    """

    def __init__(self, container, label: str, unit: str = "cells"):
        import time
        self._time = time
        self.t0 = None                      # started on the first update
        self.label = label
        self.unit = unit
        self.done = False
        self.bar = container.progress(0.0, text=f"{label} — waiting")

    def start(self, note: str = ""):
        self.t0 = self._time.time()
        self.bar.progress(0.0, text=f"{self.label} — running{('  ·  ' + note) if note else ''}")

    def __call__(self, done: int, total: int):
        if self.t0 is None:
            self.start()
        frac = (done / total) if total else 1.0
        elapsed = self._time.time() - self.t0
        eta = (elapsed / done) * (total - done) if done else None
        tail = format_eta(eta)
        self.bar.progress(min(max(frac, 0.0), 1.0),
                          text=f"{self.label} — {done}/{total} {self.unit}"
                               + (f"  ·  {tail}" if tail else ""))

    def note(self, msg: str):
        """Show what this stage is doing without adding a separate line."""
        if self.done:
            return
        if self.t0 is None:
            self.start()
        self.bar.progress(0.0, text=f"{self.label} — {msg}")

    def finish(self, note: str = ""):
        if self.t0 is None:
            self.t0 = self._time.time()
        elapsed = int(round(self._time.time() - self.t0))
        self.done = True
        self.bar.progress(1.0, text=f"{self.label} — done in {elapsed}s"
                                    + (f"  ·  {note}" if note else ""))

    def fail(self, msg: str = ""):
        self.done = True
        self.bar.progress(1.0, text=f"{self.label} — failed{('  ·  ' + msg) if msg else ''}")


class RunProgress:
    """
    Progress for a whole run: every stage is listed up front, each with its own bar.

    Listing the stages before any of them starts means the panel shows what the run
    consists of and which part is currently moving, rather than lines appearing one
    at a time with bars against some of them and not others.
    """

    def __init__(self, container, stages: list[tuple[str, str]]):
        import time
        self._time = time
        self.t0 = time.time()
        self.container = container
        self.overall = container.progress(0.0, text=f"0 of {len(stages)} stages complete")
        self.n = len(stages)
        self.bars = {}
        for key, label in stages:
            unit = "cells"
            if key.startswith("later"):
                unit = "participants"
            elif key.startswith("bayes"):
                unit = "steps"
            self.bars[key] = StepBar(container, label, unit=unit)

    def __getitem__(self, key):
        return self.bars[key]

    def _refresh(self):
        done = sum(1 for b in self.bars.values() if b.done)
        elapsed = int(round(self._time.time() - self.t0))
        self.overall.progress(done / self.n if self.n else 1.0,
                              text=f"{done} of {self.n} stages complete  ·  {elapsed}s elapsed")

    def finish_stage(self, key, note: str = ""):
        self.bars[key].finish(note)
        self._refresh()

    def fail_stage(self, key, msg: str = ""):
        self.bars[key].fail(msg)
        self._refresh()

    def finish(self):
        elapsed = int(round(self._time.time() - self.t0))
        self.overall.progress(1.0, text=f"All {self.n} stages complete  ·  {elapsed}s total")
