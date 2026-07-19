"""
Report export: a self-contained HTML report (figures embedded as base64, tables
as HTML) plus a ZIP bundle of the report, figures, and result tables.
"""

from __future__ import annotations

import base64
import io
import zipfile
from datetime import datetime

import pandas as pd


def _fig_b64(fig) -> str:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0); return base64.b64encode(buf.read()).decode("ascii")


def _fig_png(fig, dpi=300) -> bytes:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0); return buf.read()


def _tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="tbl", float_format=lambda x: f"{x:.3f}")


_CSS = """<style>
 body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:32px auto;color:#111;line-height:1.5;padding:0 18px}
 h1{font-size:22px;border-bottom:2px solid #111;padding-bottom:6px}
 h2{font-size:16px;margin-top:26px}
 .tbl{border-collapse:collapse;font-size:13px;margin:8px 0}
 .tbl td,.tbl th{border-bottom:1px solid #ccc;padding:4px 12px;text-align:right}
 .tbl th{border-bottom:1.5px solid #111}
 .muted{color:#666;font-size:12px}
 img{max-width:100%;margin:10px 0}
 .note{background:#f6f6f6;padding:10px 14px;border-left:3px solid #999;font-size:13px}
</style>"""


def build_html_report(context: dict) -> str:
    """
    context keys (all optional):
      title, subtitle, filter_report, cell_summary (DataFrames)
      results : {effector: fit_effector result dict}
      gof     : {effector: goodness_of_fit dict}
      later   : LATER result dict
      figures : {caption: matplotlib Figure}
    """
    c = context
    p = ["<html><head><meta charset='utf-8'>", _CSS, "</head><body>"]
    p.append(f"<h1>{c.get('title', 'KINARM RT analysis report')}</h1>")
    p.append(f"<p class='muted'>{c.get('subtitle', '')} &middot; generated "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")

    if isinstance(c.get("filter_report"), pd.DataFrame):
        p.append("<h2>Inclusion windows</h2>"); p.append(_tbl(c["filter_report"].round(1)))

    if isinstance(c.get("cell_summary"), pd.DataFrame):
        p.append("<h2>Distribution shape by condition</h2>")
        p.append("<p class='note'>skew / CV near 3 means near-symmetric for the spread, where a "
                 "shifted Wald cannot lift the non-decision time above the floor.</p>")
        p.append(_tbl(c["cell_summary"].round(2)))

    for eff in ("hand", "eye"):
        res = (c.get("results") or {}).get(eff)
        if not res:
            continue
        p.append(f"<h2>{eff.capitalize()}: group-level parameters by speed</h2>")
        if isinstance(res.get("group"), pd.DataFrame) and len(res["group"]):
            p.append(_tbl(res["group"].round(2)))
        cv = res.get("convergence", {})
        p.append(f"<p class='muted'>Convergence: max R-hat {cv.get('max_rhat', float('nan')):.3f}, "
                 f"divergences {cv.get('n_divergences', 0)} "
                 f"({'clean' if cv.get('converged') else 'increase sampler effort for a final run'}).</p>")
        g = (c.get("gof") or {}).get(eff)
        if g and not pd.isna(g.get("median_ks", float("nan"))):
            p.append(f"<p class='muted'>Goodness of fit: median KS {g['median_ks']:.3f}.</p>")
        if isinstance(res.get("mixture"), pd.DataFrame) and len(res["mixture"]):
            p.append(f"<h2>{eff.capitalize()}: express/regular mixture cells</h2>")
            p.append(_tbl(res["mixture"][["participant", "speed", "n", "pi",
                                          "express_mode", "reg_mode"]].round(2)))

    if c.get("later"):
        p.append("<h2>LATER model (saccades)</h2>")
        lat = c["later"]
        p.append(f"<p>Median reciprobit R&sup2; = {lat['median_r2']:.2f}. Express-dominant participants: "
                 f"{int(lat['per_participant']['express_dominant'].sum())} of "
                 f"{len(lat['per_participant'])}.</p>")
        if isinstance(lat.get("per_condition"), pd.DataFrame):
            p.append(_tbl(lat["per_condition"].round(1)))

    for caption, fig in (c.get("figures") or {}).items():
        p.append(f"<h2>{caption}</h2>"); p.append(f"<img src='data:image/png;base64,{_fig_b64(fig)}'/>")

    p.append("</body></html>")
    return "".join(p)


def build_zip_bundle(context: dict) -> bytes:
    html = build_html_report(context)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", html)
        for caption, fig in (context.get("figures") or {}).items():
            safe = caption.lower().replace(" ", "_").replace(":", "").replace("/", "_")[:60]
            z.writestr(f"figures/{safe}.png", _fig_png(fig))
        if isinstance(context.get("filter_report"), pd.DataFrame):
            z.writestr("tables/filter_report.csv", context["filter_report"].to_csv(index=False))
        if isinstance(context.get("cell_summary"), pd.DataFrame):
            z.writestr("tables/cell_summary.csv", context["cell_summary"].to_csv(index=False))
        for eff in ("hand", "eye"):
            res = (context.get("results") or {}).get(eff)
            if not res:
                continue
            for name in ("units", "group", "mixture"):
                t = res.get(name)
                if isinstance(t, pd.DataFrame) and len(t):
                    z.writestr(f"tables/{eff}_{name}.csv", t.to_csv(index=False))
            g = (context.get("gof") or {}).get(eff)
            if g and isinstance(g.get("cell"), pd.DataFrame) and len(g["cell"]):
                z.writestr(f"tables/{eff}_ks.csv", g["cell"].to_csv(index=False))
        if context.get("later") and isinstance(context["later"].get("per_cell"), pd.DataFrame):
            z.writestr("tables/later_per_cell.csv", context["later"]["per_cell"].to_csv(index=False))
    buf.seek(0)
    return buf.read()
