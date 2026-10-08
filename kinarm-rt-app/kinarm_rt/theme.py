"""
theme.py -- the app's look: translucent "liquid glass" surfaces over a soft aurora, iOS-style segmented controls,
and one moving element, the hero orbit, which draws the experiment's actual targets travelling round the ring
counter-clockwise at their real angular speeds.

Two Streamlit lessons are baked in (both cost real debugging time in v1):
  * no transform, filter or backdrop-filter on any Streamlit container — each makes the container the containing
    block for fixed-position children and breaks the fullscreen image/table overlay; the glass look comes from
    layered translucency instead, and blur is used only inside self-contained HTML;
  * entrance motion is opacity-only.
Glass panels are keyed containers: st.container(key="kp-...") gets the class st-key-kp-..., which the CSS targets.
"""
from __future__ import annotations
import streamlit as st

TOKENS = {
    "light": dict(bg="#edf1f7", a1="rgba(27,127,121,.16)", a2="rgba(217,95,2,.11)", a3="rgba(47,107,255,.13)",
                  glass="rgba(255,255,255,.60)", glass2="rgba(255,255,255,.84)", edge="rgba(255,255,255,.95)",
                  line="rgba(14,26,43,.09)", ink="#0e1a2b", ink2="#4b5a73", ink3="#7a879b", accent="#2f6bff",
                  shadow="0 1px 1px rgba(14,26,43,.04), 0 14px 34px rgba(14,26,43,.08)",
                  rec="#1f9d6b", sup="#2f6bff", dia="#6b5fd3", dep="#b7791f", bad="#c2413a", figbg="#ffffff"),
    "dark": dict(bg="#070b13", a1="rgba(27,127,121,.24)", a2="rgba(217,95,2,.15)", a3="rgba(110,155,255,.17)",
                 glass="rgba(22,29,45,.58)", glass2="rgba(28,37,57,.82)", edge="rgba(255,255,255,.09)",
                 line="rgba(255,255,255,.08)", ink="#e8edf6", ink2="#a3aec2", ink3="#6f7b90", accent="#6e9bff",
                 shadow="0 1px 1px rgba(0,0,0,.3), 0 18px 40px rgba(0,0,0,.35)",
                 rec="#3ccb8e", sup="#6e9bff", dia="#9a8cf0", dep="#e2a74a", bad="#ff7a70", figbg="#f7f8fb"),
}

FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif'


def current_theme() -> str:
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None) in ("light", "dark"): return t.type
    except Exception:
        pass
    return "light"


def inject(theme: str) -> None:
    t = TOKENS[theme]
    st.markdown(f"<style>{_css(t)}{CARD_CSS}</style>", unsafe_allow_html=True)


def _css(t: dict) -> str:
    return f"""
:root {{ --k-bg:{t['bg']}; --k-glass:{t['glass']}; --k-glass2:{t['glass2']}; --k-edge:{t['edge']}; --k-line:{t['line']};
  --k-ink:{t['ink']}; --k-ink2:{t['ink2']}; --k-ink3:{t['ink3']}; --k-accent:{t['accent']}; --k-shadow:{t['shadow']};
  --k-rec:{t['rec']}; --k-sup:{t['sup']}; --k-dia:{t['dia']}; --k-dep:{t['dep']}; --k-bad:{t['bad']}; --k-figbg:{t['figbg']};
  --k-font:{FONT}; }}
.stApp {{ background:
    radial-gradient(1100px 620px at 6% -12%, {t['a1']}, transparent 62%),
    radial-gradient(900px 620px at 102% -4%, {t['a3']}, transparent 58%),
    radial-gradient(900px 560px at 58% 112%, {t['a2']}, transparent 62%), var(--k-bg);
  background-attachment: fixed; color: var(--k-ink); }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input, .stApp textarea
  {{ font-family: var(--k-font); }}
.stApp h1, .stApp h2, .stApp h3 {{ letter-spacing:-0.015em; color:var(--k-ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }}
@keyframes kFade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}

/* glass panels: keyed containers, no blur / transform on the container itself */
[class*="st-key-kp-"] {{ background: var(--k-glass); border: 1px solid var(--k-line); border-radius: 22px;
  box-shadow: var(--k-shadow), inset 0 1px 0 var(--k-edge); padding: 22px 24px 22px; gap: 0.75rem; animation: kFade .35s ease-out backwards; }}
[class*="st-key-kp-"] h3, [class*="st-key-kq-"] h3, [class*="st-key-kd-"] h3 {{ font-size: 1.22rem; font-weight: 680; margin: 0 0 2px; padding: 0; line-height: 1.3; }}
[class*="st-key-k"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
[class*="st-key-k"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0.45rem; }}
[class*="st-key-k"] [data-testid="stMarkdownContainer"] > :last-child {{ margin-bottom: 0 !important; }}
[class*="st-key-k"] [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"]) {{ margin-bottom: 0 !important; }}
[class*="st-key-kq-"] {{ background: var(--k-glass2); border: 1px solid var(--k-line); border-radius: 16px;
  box-shadow: inset 0 1px 0 var(--k-edge); padding: 16px 18px 16px; gap: 0.5rem; }}
[class*="st-key-kd-"] {{ background: color-mix(in srgb, var(--k-dep) 9%, var(--k-glass2)); border: 1px solid color-mix(in srgb, var(--k-dep) 35%, transparent);
  border-radius: 16px; padding: 16px 18px 16px; gap: 0.5rem; }}

/* segmented controls as iOS segments */
[data-testid="stButtonGroup"] {{ background: var(--k-glass2); border: 1px solid var(--k-line); border-radius: 999px; padding: 3px;
  box-shadow: inset 0 1px 0 var(--k-edge); width: fit-content; }}
[data-testid="stButtonGroup"] button {{ border: none !important; background: transparent; border-radius: 999px !important;
  color: var(--k-ink2); font-weight: 560; padding: 5px 14px; min-height: 32px; transition: background-color .18s, color .18s; }}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"], [data-testid="stButtonGroup"] button[kind="pillsActive"]
  {{ background: var(--k-ink) !important; color: var(--k-bg) !important; box-shadow: 0 2px 10px rgba(0,0,0,.12); }}
[data-testid="stButtonGroup"] button:focus-visible {{ outline: 2px solid var(--k-accent); outline-offset: 2px; }}

/* buttons */
.stApp button[kind="primary"] {{ background: var(--k-accent); border: none; border-radius: 999px; font-weight: 620;
  box-shadow: 0 6px 18px color-mix(in srgb, var(--k-accent) 35%, transparent); }}
.stApp button[kind="secondary"] {{ border-radius: 999px; border: 1px solid var(--k-line); background: var(--k-glass2); }}
.stApp button:focus-visible {{ outline: 2px solid var(--k-accent); outline-offset: 2px; }}

/* data, figures, expanders */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid var(--k-line); }}
[data-testid="stImage"] img {{ border-radius: 12px; background: var(--k-figbg); }}
[data-testid="stExpander"] details {{ border-radius: 16px; border: 1px solid var(--k-line); background: var(--k-glass); }}
[data-testid="stFileUploaderDropzone"] {{ border-radius: 18px; border: 1.5px dashed color-mix(in srgb, var(--k-accent) 40%, var(--k-line));
  background: var(--k-glass2); }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}

/* custom HTML pieces */
.k-brand {{ display:flex; align-items:center; gap:10px; font-weight:680; font-size:1.05rem; color:var(--k-ink); }}
.k-brand small {{ font-weight:500; color:var(--k-ink3); font-size:.85rem; }}
.k-hero {{ position:relative; overflow:hidden; border-radius: 28px; padding: 30px 32px 28px; min-height: 248px;
  background: linear-gradient(135deg, var(--k-glass2), var(--k-glass)); border:1px solid var(--k-line);
  box-shadow: var(--k-shadow), inset 0 1px 0 var(--k-edge); animation: kFade .5s ease-out backwards; }}
.k-hero .k-copy {{ position:relative; z-index:1; max-width: min(700px, calc(100% - 290px)); }}
.k-hero h1 {{ font-size: 2.0rem; line-height:1.14; margin: 4px 0 10px; font-weight: 720; padding:0; }}
.k-hero p {{ color: var(--k-ink2); margin: 0 0 16px; font-size: 1.0rem; line-height: 1.55; }}
.k-hero .k-orbit {{ position:absolute; z-index:0; pointer-events:none; }}
@media (max-width: 900px) {{ .k-hero .k-copy {{ max-width: 100%; }} .k-hero .k-orbit {{ opacity:.22; }} }}
.k-kicker {{ display:inline-flex; align-items:center; gap:8px; color: var(--k-ink2); font-weight: 600; font-size:.92rem; }}
.k-kicker i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.k-chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
.k-chip {{ display:inline-flex; align-items:baseline; gap:6px; padding: 6px 12px; border-radius: 999px; background: var(--k-glass2);
  border:1px solid var(--k-line); font-size:.88rem; color: var(--k-ink2); box-shadow: inset 0 1px 0 var(--k-edge); }}
.k-chip b {{ color: var(--k-ink); font-weight: 680; font-variant-numeric: tabular-nums; }}
.k-badge {{ display:inline-block; vertical-align: middle; margin-left: 4px; padding: 2px 10px; border-radius: 999px; font-size:.78rem; font-weight: 650; letter-spacing:.01em;
  border:1px solid currentColor; background: color-mix(in srgb, currentColor 10%, transparent); }}
.k-badge.recommended {{ color: var(--k-rec); }} .k-badge.supporting {{ color: var(--k-sup); }}
.k-badge.diagnostic {{ color: var(--k-dia); }} .k-badge.deprecated {{ color: var(--k-dep); }}
.k-badge.main {{ color: var(--k-rec); }} .k-badge.method_a {{ color: var(--k-ink3); }} .k-badge.compare {{ color: var(--k-accent); }}
.k-sub {{ color: var(--k-ink2); font-size: .95rem; line-height: 1.5; margin: 2px 0 8px; }}
.k-muted {{ color: var(--k-ink3); font-size: .86rem; }}
.k-steps {{ list-style:none; padding:0; margin: 4px 0 0; }}
.k-steps li {{ display:grid; grid-template-columns: 22px 1fr auto; gap: 10px; padding: 7px 2px; border-bottom: 1px solid var(--k-line);
  font-size:.93rem; color: var(--k-ink); }}
.k-steps li:last-child {{ border-bottom: none; }}
.k-steps .s {{ font-weight:700; text-align:center; }} .k-steps .t {{ color: var(--k-ink3); font-variant-numeric: tabular-nums; }}
.k-steps .done .s {{ color: var(--k-rec); }} .k-steps .running .s {{ color: var(--k-accent); }} .k-steps .failed .s, .k-steps .failed {{ color: var(--k-bad); }}
.k-steps .skipped .s, .k-steps .blocked .s, .k-steps .cancelled .s, .k-steps .pending .s {{ color: var(--k-ink3); }}
.k-steps .detail {{ color: var(--k-ink3); font-size:.84rem; }}
.k-orbit .ring {{ fill:none; stroke: var(--k-ink3); stroke-opacity:.45; stroke-width:1.4; stroke-dasharray: 2 5; }}
.k-orbit .orb {{ transform-origin: 100px 100px; transform-box: view-box; animation-name: kOrbit; animation-timing-function: linear;
  animation-iteration-count: infinite; }}
.k-orbit .glow {{ filter: blur(3px); opacity:.55; }}
@keyframes kOrbit {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(-360deg); }} }}
@keyframes kPulse {{ 0%, 100% {{ opacity:.25; }} 50% {{ opacity:.75; }} }}
.k-orbit .pulse {{ animation: kPulse 2.8s ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{ .k-orbit .orb, .k-orbit .pulse, [data-testid="stImage"] img {{ animation: none; }} [class*="st-key-kp-"], .k-hero {{ animation: none; }} }}
"""


def orbit_svg(speeds, colours, size: int = 360) -> str:
    """The task, drawn: moving targets travel counter-clockwise round the ring at their real angular speeds (one
    revolution takes 360/speed seconds), spread out in phase; a stationary target stays put on the left of the ring. Sized and placed with explicit
    pixels at the right of the hero, so it never sits behind the text and renders the same in every browser engine
    (WebKit sizes a viewBox-only SVG to its container, which is what blew the ring up behind the text in 2.0)."""
    style = f"width:{size}px;height:{size}px;right:-{size // 5}px;top:50%;margin-top:-{size // 2}px"
    parts = [f'<svg class="k-orbit" width="{size}" height="{size}" viewBox="0 0 200 200" style="{style}" aria-hidden="true">',
             '<circle cx="100" cy="100" r="78" class="ring"/>']
    moving = [s for s in speeds if s > 0]
    for i, s in enumerate(speeds):
        c = colours.get(s, "#888")
        if s == 0:   # the stationary target sits on the left of the ring, where the whole dot is visible
            parts.append(f'<circle cx="22" cy="100" r="9" fill="{c}" class="glow pulse"/><circle cx="22" cy="100" r="5.5" fill="{c}"/>'); continue
        period = 360.0 / s; delay = -period * (i / max(len(moving), 1))
        parts.append(f'<g class="orb" style="animation-duration:{period:.2f}s;animation-delay:{delay:.2f}s">'
                     f'<circle cx="178" cy="100" r="7.5" fill="{c}" class="glow"/><circle cx="178" cy="100" r="5.5" fill="{c}"/></g>')
    parts.append("</svg>")
    return "".join(parts)


def orbit_compare_svg(inner, outer, colours, inner_accent: str, outer_accent: str, size: int = 360) -> str:
    """Both experiments at once: Experiment 1's targets on the inner ring (its stationary target on the left),
    Experiment 2's on the outer ring, each moving counter-clockwise at its real angular speed."""
    style = f"width:{size}px;height:{size}px;right:-{size // 5}px;top:50%;margin-top:-{size // 2}px"
    parts = [f'<svg class="k-orbit" width="{size}" height="{size}" viewBox="0 0 200 200" style="{style}" aria-hidden="true">',
             f'<circle cx="100" cy="100" r="86" class="ring" style="stroke:{outer_accent};stroke-opacity:.55"/>',
             f'<circle cx="100" cy="100" r="54" class="ring" style="stroke:{inner_accent};stroke-opacity:.65"/>']
    for ring, speeds, rr, dot in (("inner", inner, 54, 4.6), ("outer", outer, 86, 5.4)):
        moving = [x for x in speeds if x > 0]
        for i, sp in enumerate(speeds):
            c = colours.get(sp, "#888"); cx = 100 + rr
            if sp == 0:
                parts.append(f'<circle cx="{100 - rr}" cy="100" r="{dot + 2.5}" fill="{c}" class="glow pulse"/>'
                             f'<circle cx="{100 - rr}" cy="100" r="{dot}" fill="{c}"/>'); continue
            period = 360.0 / sp; delay = -period * ((i + (0.5 if ring == "outer" else 0)) / max(len(moving), 1))
            parts.append(f'<g class="orb" style="animation-duration:{period:.2f}s;animation-delay:{delay:.2f}s">'
                         f'<circle cx="{cx}" cy="100" r="{dot + 2}" fill="{c}" class="glow"/><circle cx="{cx}" cy="100" r="{dot}" fill="{c}"/></g>')
    parts.append("</svg>")
    return "".join(parts)


def hero(kicker: str, accent: str, title: str, body: str, chips: list, speeds, colours, orbit: str | None = None) -> None:
    chip_html = "".join(f'<span class="k-chip">{label} <b>{value}</b></span>' for label, value in chips)
    st.markdown(f'<div class="k-hero">{orbit or orbit_svg(speeds, colours)}<div class="k-copy"><div class="k-kicker">'
                f'<i style="background:{accent}"></i>{kicker}</div><h1>{title}</h1><p>{body}</p>'
                f'<div class="k-chips">{chip_html}</div></div></div>', unsafe_allow_html=True)


def badge(status: str, label: str) -> str:
    return f'<span class="k-badge {status}">{label}</span>'


CARD_CSS = """
/* button rows: buttons sit together with one consistent gap instead of spreading across columns */
[class*="st-key-row-"] { gap: 0.5rem !important; flex-wrap: wrap; align-items: center; }
[class*="st-key-row-"] [data-testid="stElementContainer"], [class*="st-key-row-"] > div { width: auto !important; flex: 0 0 auto !important; }
[class*="st-key-kp-pane-"] [data-testid="stImage"] img { width: 100% !important; height: auto !important; }
.k-facts { margin: 6px 0 4px; padding-left: 1.1rem; color: var(--k-ink); }
.k-facts li { margin: 2px 0; line-height: 1.5; font-variant-numeric: tabular-nums; }
/* fallbacks first for older WebKit (macOS 12) without color-mix(); modern engines use the rules in _css() */
@supports not (background: color-mix(in srgb, red 50%, blue)) {
  [class*="st-key-kd-"] { background: rgba(183,121,31,.08); border-color: rgba(183,121,31,.35); }
  .k-badge { background: transparent; }
  [data-testid="stFileUploaderDropzone"] { border-color: rgba(47,107,255,.4); }
}
[data-testid="stImage"] img { animation: kFade .45s ease-out both; }
[class*="st-key-kq-"] { transition: border-color .25s ease, box-shadow .25s ease, background-color .25s ease; }
[class*="st-key-kq-card-"]:hover { border-color: color-mix(in srgb, var(--k-accent) 35%, var(--k-line));
  box-shadow: 0 10px 28px rgba(14,26,43,.10), inset 0 1px 0 var(--k-edge); }
.stApp button { transition: background-color .18s ease, border-color .18s ease, box-shadow .18s ease, color .18s ease; }
.stApp button[kind="secondary"]:hover { border-color: color-mix(in srgb, var(--k-accent) 45%, var(--k-line)); }
.stApp button[kind="primary"]:hover { box-shadow: 0 8px 24px color-mix(in srgb, var(--k-accent) 45%, transparent); }
.k-group-chips { display:flex; flex-wrap:wrap; gap:6px; margin: 4px 0 2px; }
.k-group-chips span { font-size:.8rem; color: var(--k-ink2); padding: 3px 10px; border-radius: 999px; background: var(--k-glass2);
  border: 1px solid var(--k-line); }
.k-group-chips b { color: var(--k-ink); font-variant-numeric: tabular-nums; margin-left: 4px; }
[class*="st-key-kp-doc"] [data-testid="stMarkdownContainer"] p, [class*="st-key-kp-doc"] [data-testid="stMarkdownContainer"] li { max-width: 82ch; }
/* while a page redraws, fade what is about to be replaced out of the way instead of leaving it half-visible */
[data-stale="true"] { opacity: .12 !important; transition: opacity .12s ease !important; }
[class*="st-key-kp-doc"] h1 { font-size: 1.6rem; margin-top: .4rem; }
[class*="st-key-kp-doc"] h2 { font-size: 1.3rem; margin-top: 1.2rem; }
[class*="st-key-kp-doc"] h3 { font-size: 1.1rem; margin-top: 1rem; }
[class*="st-key-kp-doc"] table { border-collapse: collapse; font-size: .9rem; margin: .6rem 0 1rem; }
[class*="st-key-kp-doc"] th, [class*="st-key-kp-doc"] td { border: 1px solid var(--k-line); padding: 5px 9px; }
[class*="st-key-kp-doc"] th { background: var(--k-glass2); }
[class*="st-key-kp-doc"] code { font-size: .86em; }
[class*="st-key-kq-card-"] { min-height: 318px; justify-content: space-between; }
[class*="st-key-kq-card-"] [data-testid="stImage"] img { height: 178px !important; width: 100% !important; object-fit: contain; padding: 6px; }
.k-title { font-weight: 650; color: var(--k-ink); line-height: 1.32; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden; min-height: 2.64em; margin: 2px 0 6px; }
.k-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.k-count { color: var(--k-ink3); font-size: .88rem; font-variant-numeric: tabular-nums; }
.k-pane-title { font-weight: 680; font-size: 1.05rem; color: var(--k-ink); margin: 0; }
"""
