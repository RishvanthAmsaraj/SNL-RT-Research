"""
results.py -- read what the scripts wrote: the committed reference results shipped with the app, or a run folder.
Only presentation happens here (finding files, rendering PDF previews, packaging a zip, a few headline numbers).
"""
from __future__ import annotations
import io, os, tempfile, zipfile
from dataclasses import dataclass
import numpy as np, pandas as pd

APP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(APP, "reference_results")
_PREVIEWS = os.path.join(tempfile.gettempdir(), "kinarm_rt_previews")


@dataclass
class ResultSet:
    label: str
    source: str               # "reference" | "run"
    folders: tuple

    def find(self, name: str) -> str | None:
        for f in self.folders:
            p = os.path.join(f, name)
            if os.path.exists(p): return p
        return None

    def table(self, name: str) -> pd.DataFrame | None:
        p = self.find(name)
        if not p: return None
        try: return pd.read_csv(p)
        except Exception: return None

    def tables(self) -> list[str]:
        seen = []
        for f in self.folders:
            if os.path.isdir(f):
                for n in sorted(os.listdir(f)):
                    if n.endswith(".csv") and n not in seen and not n.startswith(("pooled_data",)): seen.append(n)
        return seen

    def image_stems(self) -> list[str]:
        """Every figure in these results (one entry per figure, whether it exists as PNG, PDF or both)."""
        seen = []
        for f in self.folders:
            if os.path.isdir(f):
                for n in sorted(os.listdir(f)):
                    if n.endswith((".png", ".pdf")):
                        stem = n.rsplit(".", 1)[0]
                        if stem not in seen: seen.append(stem)
        return seen

    def image(self, stem: str) -> str | None:
        """PNG for display; if the script wrote only a PDF, a preview is rendered (the PDF itself is untouched)."""
        png = self.find(stem + ".png")
        if png: return png
        pdf = self.find(stem + ".pdf")
        if not pdf: return None
        try:
            import pymupdf
            os.makedirs(_PREVIEWS, exist_ok=True)
            out = os.path.join(_PREVIEWS, f"{abs(hash(pdf))}_{int(os.path.getmtime(pdf))}.png")
            if not os.path.exists(out):
                with pymupdf.open(pdf) as doc: doc[0].get_pixmap(dpi=130).save(out)
            return out
        except Exception:
            return None

    def thumb(self, stem: str, width: int = 560) -> str | None:
        """A small JPEG of the figure for galleries (the full image stays for the figure page and downloads).
        Committed results ship these pre-made in _thumbs/; for runs they are made once and cached."""
        for f in self.folders:
            t = os.path.join(f, "_thumbs", f"{stem}.w{width}.jpg")
            if os.path.exists(t): return t
        src = self.image(stem)
        if not src: return None
        try:
            os.makedirs(_PREVIEWS, exist_ok=True)
            out = os.path.join(_PREVIEWS, f"th_{abs(hash(src))}_{int(os.path.getmtime(src))}_{width}.jpg")
            if not os.path.exists(out): make_thumb(src, out, width)
            return out
        except Exception:
            return src

    def zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in self.folders:
                if not os.path.isdir(f): continue
                for root, _, files in os.walk(f):
                    for n in files:
                        if n.endswith((".csv", ".png", ".pdf", ".log", ".txt", ".json", ".md")) and not n.startswith("pooled_data"):
                            p = os.path.join(root, n); z.write(p, os.path.relpath(p, os.path.dirname(f)))
        return buf.getvalue()


GROUP_OF_SET = {"E1": "Results", "E2": "Results", "E1_later": "Deprecated models", "E2_later": "Deprecated models",
                "E1_two_boundary": "Deprecated models", "E2_validation": "Validation", "E1_archive": "Earlier versions",
                "compare": "Comparison"}
SETS_FOR = {"E1": ("E1", "E1_later", "E1_two_boundary", "E1_archive"), "E2": ("E2", "E2_later", "E2_validation")}
DOCS_FOR = {"E1": (("Experiment 1", "E1_docs"), ("Deprecated models", "deprecated_docs"),
                   ("Earlier versions and history", "archive_docs"), ("Repository", "repo_docs")),
            "E2": (("Experiment 2", "E2_docs"), ("Deprecated models", "deprecated_docs"), ("Repository", "repo_docs")),
            "CMP": (("Comparison", "E2_docs"), ("Repository", "repo_docs"))}


def make_thumb(src: str, out: str, width: int) -> None:
    from PIL import Image
    with Image.open(src) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA"); bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[-1]); im = bg
        else:
            im = im.convert("RGB")
        if im.width > width: im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
        im.save(out, "JPEG", quality=86, optimize=True, progressive=True)


def reference(exp_id: str) -> ResultSet:
    folders = tuple(os.path.join(REF, k) for k in SETS_FOR.get(exp_id, (exp_id,)) if os.path.isdir(os.path.join(REF, k)))
    return ResultSet("Lab results (committed)", "reference", folders)


def group_of(rs: ResultSet, name: str) -> str:
    """Which part of the repository a file in these results comes from (for grouping tables)."""
    if rs.source == "run": return "Your run"
    for f in rs.folders:
        if os.path.exists(os.path.join(f, name)): return GROUP_OF_SET.get(os.path.basename(f), "Results")
    return "Results"


def _manifest() -> dict:
    try:
        import json
        with open(os.path.join(APP, "pipelines", "MANIFEST.json")) as f: return json.load(f)
    except (OSError, ValueError):
        return {}


def original_figure_files(rs: ResultSet) -> int:
    """How many figure files (PNG and PDF) these results really hold -- previews made for display are not counted."""
    if rs.source == "run":
        return sum(1 for f in rs.folders if os.path.isdir(f) for n in os.listdir(f) if n.endswith((".png", ".pdf")))
    ref = _manifest().get("reference", {})
    return sum(1 for f in rs.folders for rec in ref.get(os.path.basename(f), []) if rec["file"].endswith((".png", ".pdf")))


def documents(key: str) -> list:
    """[(group, file name, path)] for an experiment ("E1", "E2") or the comparison ("CMP")."""
    out = []
    for group, folder in DOCS_FOR.get(key, ()):
        d = os.path.join(REF, folder)
        if not os.path.isdir(d): continue
        for n in sorted(os.listdir(d)):
            if not n.endswith((".md", ".pdf")): continue
            if key == "CMP" and folder == "E2_docs" and n not in ("P2_Results_and_P1_Comparison.md", "FIGURE_GUIDE.md"): continue
            if key == "E2" and folder == "deprecated_docs" and "two" in n.lower(): continue    # two-boundary is Experiment 1 only
            out.append((group, n, os.path.join(d, n)))
    return out


def compare_reference() -> ResultSet:
    return ResultSet("Lab results (committed)", "reference", (os.path.join(REF, "compare"),))


def from_run(run_dir: str) -> ResultSet:
    return ResultSet("Your run", "run", (run_dir,))


# --------------------------------------------------------------------------- headline numbers (display only)
def headline(rs: ResultSet, speeds) -> dict:
    out = {}
    b = rs.table("Bayesian_hrt_fits.csv")
    if b is not None and {"spd", "t0"} <= set(b.columns):
        out["hand_t0"] = {int(s): float(b[b.spd == s].t0.mean()) for s in speeds if (b.spd == s).any()}
        out["hand_floored"] = int(((b.t0 - 130).abs() < 2).sum()); out["hand_cells"] = len(b)
        out["participants"] = int(b.pid.nunique())
        try:
            from scipy.stats import friedmanchisquare
            w = b.pivot_table(index="pid", columns="spd", values="t0")[[s for s in speeds]].dropna()
            out["friedman_p"] = float(friedmanchisquare(*[w[s] for s in speeds]).pvalue)
        except Exception:
            pass
    n = rs.table("Bayesian_srt_ndt.csv")
    if n is not None and {"t0_lo95", "t0_hi95", "min_srt_ms"} <= set(n.columns):
        fl = n.t0_lo95 <= 71; ce = n.t0_hi95 >= n.min_srt_ms - 2
        out["sacc"] = {"n": len(n), "floor": int(fl.sum()), "ceiling": int(ce.sum()), "free": int((~fl & ~ce).sum()),
                       "range": (int(n.t0_ms.min()), int(n.t0_ms.max()))}
    lat = rs.table("LATER_fits.csv")
    if lat is not None and "reciprobit_r2" in lat:
        out["later_r2"] = float(lat.reciprobit_r2.median())
    return out


def pdf_pages(path: str) -> int:
    try:
        import pymupdf
        with pymupdf.open(path) as doc: return doc.page_count
    except Exception:
        return 0


def pdf_preview(path: str, page: int = 0) -> str | None:
    """PNG of one page of a PDF document, for display."""
    try:
        import pymupdf
        os.makedirs(_PREVIEWS, exist_ok=True)
        out = os.path.join(_PREVIEWS, f"doc_{abs(hash(path))}_{int(os.path.getmtime(path))}_{page}.png")
        if not os.path.exists(out):
            with pymupdf.open(path) as doc: doc[page].get_pixmap(dpi=110).save(out)
        return out
    except Exception:
        return None
