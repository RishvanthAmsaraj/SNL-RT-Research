"""
sync_pipelines.py -- copy the repository's analysis scripts and committed results into the app, unchanged.

The app never re-implements an analysis. It runs the repository's own scripts, so every number and figure it produces
is the one the scripts produce. This tool keeps that promise checkable:

    python tools/sync_pipelines.py            # copy scripts, results and documents from the repo, write MANIFEST.json
    python tools/sync_pipelines.py --check    # exit 1 if any vendored script differs from its repo source

Vendored scripts land flat in pipelines/<pipeline>/ (each pipeline's scripts expect to sit in one folder, as when run by
hand). Everything the repository holds as results -- the current figures and tables of both experiments, the
deprecated models, the earlier pipeline versions and working iterations, and the documents -- lands in
reference_results/<set>/, so the app can show all of it without running anything. Earlier versions reuse file names,
so their copies are prefixed with the version (v3__DDM_summary.png). Vincentile figures exist only as PDF; a PNG
preview is rendered for display (the PDF itself is copied untouched). Identical files are kept once.
"""
from __future__ import annotations
import glob, hashlib, json, os, re, shutil, sys, datetime

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
FIG = ("*.png", "*.pdf")

PIPELINES = {
    "experiment1": [("Current Pipeline/Code/" + d, "*.py") for d in ["DDM", "Bayesian", "NDT", "SRT Analysis", "Vincentile", "Supplementary"]],
    "experiment2": [("Paradigm 2 Pipeline/Code/" + d, "*.py") for d in ["DDM", "Bayesian", "NDT", "SRT Analysis", "Vincentile", "Supplementary"]]
                   + [("Paradigm 2 Pipeline/Code", "build_pooled_data_P2.py")],
    "later_e1": [("Deprecated Pipelines/LATER/Experiment 1", "LATER_analysis.py")],
    "later_e2": [("Deprecated Pipelines/LATER/Experiment 2", "LATER_analysis.py")],
}
REFERENCE = {
    "E1": [("Current Pipeline/Code/" + d, "*.csv") for d in ["DDM", "Bayesian", "SRT Analysis", "Supplementary"]]
          + [("Current Pipeline/Figures/" + d, p) for d in ["DDM", "Bayesian", "NDT", "SRT Analysis", "Vincentile", "Supplementary"] for p in FIG],
    "E2": [("Paradigm 2 Pipeline/Code/" + d, "*.csv") for d in ["DDM", "Bayesian", "SRT Analysis", "Supplementary"]]
          + [("Paradigm 2 Pipeline/Figures/" + d, p) for d in ["DDM", "Bayesian", "NDT", "SRT Analysis", "Vincentile", "Supplementary"] for p in FIG],
    "E2_validation": [("Paradigm 2 Pipeline/Code/Validation", "*.csv")],
    "E1_later": [("Deprecated Pipelines/LATER/Experiment 1", p) for p in ("*.csv",) + FIG],
    "E2_later": [("Deprecated Pipelines/LATER/Experiment 2", p) for p in ("*.csv",) + FIG],
    "E1_two_boundary": [("Deprecated Pipelines/Two-boundary DDM/Experiment 1/results", p) for p in ("*.csv",) + FIG],
    "compare": [("Paradigm 2 Pipeline/Figures/Comparison", p) for p in FIG] + [("Paradigm 2 Pipeline/Code/Comparison", "*.csv")],
    "E1_docs": [("Current Pipeline/Documents", p) for p in ("*.md", "*.pdf")],
    "E2_docs": [("Paradigm 2 Pipeline/Documents", "*.md")],
    "repo_docs": [("", "README.md"), ("", "CHANGELOG.md"), ("", "DEVELOPMENT_HISTORY.md"), ("kinarm-rt-app", "FIGURE_EXPLANATIONS.md")],
}
RENAMED = {   # sets whose files need new names (several are called README.md)
    "deprecated_docs": {"LATER_README.md": "Deprecated Pipelines/LATER/README.md",
                        "Two-boundary_README.md": "Deprecated Pipelines/Two-boundary DDM/README.md",
                        "VERDICT_two_boundary_vs_single.md": "Deprecated Pipelines/Two-boundary DDM/Experiment 1/results/VERDICT_two_boundary_vs_single.md",
                        "README_TWO_BOUNDARY.md": "Deprecated Pipelines/Two-boundary DDM/Experiment 1/code/README_TWO_BOUNDARY.md"},
}
# earlier pipeline versions (all Experiment 1 data) and the working iterations: copied whole, names prefixed
ARCHIVE = [("v1", "Deprecated Pipelines/Deprecated Ver 1"), ("v2", "Deprecated Pipelines/Deprecated Ver 2"),
           ("v2_5", "Deprecated Pipelines/Deprecated Ver 2.5"), ("v3", "Deprecated Pipelines/Deprecated Ver 3"),
           ("wi", "Working Iterations")]
SKIP = re.compile(r"(^pooled_data|^example_ddm_data)", re.I)
# De-identified participant codes (CMT001..., CIR001...). A table that carries these is
# per-participant data and must never be shipped in the app, even if it lives in a
# gitignored folder in the repository.
PARTICIPANT_ID = re.compile(r"\b(CMT[0-9]{3,4}|CIR[0-9]{3})\b")


def _is_participant_data(path: str) -> bool:
    """True when a CSV table holds per-participant rows (de-identified codes)."""
    if not path.lower().endswith(".csv"):
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.read(65536)
    except OSError:
        return False
    return bool(PARTICIPANT_ID.search(head))
# Figures the Experiment 1 correction (October 2026) fixed: their earlier copies show the mistake, so they are kept as
# deprecated. Every other version-3 figure differs from the current one only by its title, so it is the same figure.
CORRECTED = {"DDM_summary", "Bayesian_summary", "Bayesian_srt_ndt", "SRT_fixedt0_sensitivity", "NDT_barchart_bayesian",
             "SRT_identifiability", "why_saccadic_t0_floors", "ddm_srt_0_degs"}
SAME_FIGURE_DIFF = 1.75        # mean absolute grey-level difference (0-255) below which two renders are the same figure


def _render(path: str):
    """Grey-scale image of a figure file (PNG, or the first page of a PDF) for comparing two versions."""
    from PIL import Image
    if path.lower().endswith(".pdf"):
        import pymupdf, io
        with pymupdf.open(path) as doc: data = doc[0].get_pixmap(dpi=60).tobytes("png")
        return Image.open(io.BytesIO(data)).convert("L")
    return Image.open(path).convert("L")


def same_figure(a: str, b: str) -> bool:
    import numpy as np
    A, B = _render(a), _render(b)
    if abs(A.width / A.height - B.width / B.height) > 0.01 * (B.width / B.height): return False
    w = 480; h = max(1, round(w * A.height / A.width))
    return float(np.abs(np.asarray(A.resize((w, h)), float) - np.asarray(B.resize((w, h)), float)).mean()) < SAME_FIGURE_DIFF


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def collect(spec):
    out = {}
    for folder, pattern in spec:
        for src in sorted(glob.glob(os.path.join(REPO, folder, pattern))):
            if _is_participant_data(src):
                continue
            name = os.path.basename(src)
            if name in out and sha(out[name]) != sha(src):
                sys.exit(f"name clash: {name} comes from two different files ({out[name]} and {src})")
            out[name] = src
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def collect_archive(seen_hashes: set):
    """-> (results {name: src}, docs {name: src}). Prefix = version tag; a parent-folder slug is added only where a
    version reuses a file name. Files identical to something already shipped are skipped."""
    res, docs = {}, {}
    for tag, folder in ARCHIVE:
        base = os.path.join(REPO, folder)
        found = []
        for d, _, fs in os.walk(base):
            for f in sorted(fs):
                if SKIP.search(f) or not f.lower().endswith((".png", ".pdf", ".csv", ".md")): continue
                found.append(os.path.join(d, f))
        counts = {}
        for p in found: counts[os.path.basename(p)] = counts.get(os.path.basename(p), 0) + 1
        for p in sorted(found):
            if _is_participant_data(p):
                continue
            name = os.path.basename(p); rel_dir = os.path.relpath(os.path.dirname(p), base)
            is_doc = name.lower().endswith(".md") or (name.lower().endswith(".pdf") and "Documents" in rel_dir.split(os.sep))
            h = sha(p)
            if h in seen_hashes: continue
            seen_hashes.add(h)
            parts = [tag] + ([_slug(rel_dir)] if (counts[name] > 1 or tag == "wi") and rel_dir != "." else []) + [name]
            target = "__".join(parts)
            (docs if is_doc else res)[target] = p
    return res, docs


def drop_current_copies(arch: dict):
    """Keep an earlier-version figure only if it really is older: it shows a mistake that was corrected, or it differs
    from the current figure. Copies of the current figure are dropped and recorded (they are accounted for, not lost)."""
    cur_dir = {}
    for key in ("E1", "E1_later", "E1_two_boundary"):
        d = os.path.join(APP, "reference_results", key)
        for f in os.listdir(d):
            if f.endswith((".png", ".pdf")): cur_dir.setdefault(f.rsplit(".", 1)[0], {})[f.rsplit(".", 1)[1]] = os.path.join(d, f)
    by_stem = {}
    for target, src in arch.items():
        if target.endswith((".png", ".pdf")): by_stem.setdefault(target.rsplit(".", 1)[0], {})[target] = src
    keep, dups = dict(arch), []
    for stem, files in by_stem.items():
        tag, base = stem.split("__")[0], stem.split("__")[-1]
        if base not in cur_dir or base in CORRECTED: continue
        if tag == "v3":
            same = True
        else:
            # compare like with like: both PDFs rendered the same way when both exist, otherwise the PNGs
            mine_pdf = next((s for t, s in files.items() if t.endswith(".pdf")), None)
            mine_png = next((s for t, s in files.items() if t.endswith(".png")), None)
            if mine_pdf and cur_dir[base].get("pdf"): mine, theirs = mine_pdf, cur_dir[base]["pdf"]
            else: mine, theirs = (mine_png or mine_pdf), (cur_dir[base].get("png") or cur_dir[base].get("pdf"))
            same = same_figure(mine, theirs)
        if same:
            for t, src in files.items():
                keep.pop(t, None)
                dups.append({"source": os.path.relpath(src, REPO).replace(os.sep, "/"), "sha256": sha(src), "same_as": base})
    return keep, dups


def pdf_previews(folder: str) -> int:
    """PNG preview for every figure PDF that has no PNG twin (display only)."""
    try:
        import pymupdf
    except ImportError:
        return 0
    n = 0
    for pdf in glob.glob(os.path.join(folder, "*.pdf")):
        png = pdf[:-4] + ".png"
        if not os.path.exists(png):
            with pymupdf.open(pdf) as doc: doc[0].get_pixmap(dpi=130).save(png); n += 1
    return n


def make_thumbs(folder: str, width: int = 560) -> int:
    """Gallery-sized JPEGs in <set>/_thumbs/ (display only; the figures themselves are untouched)."""
    sys.path.insert(0, APP)
    from kinarm_rt.engine.results import make_thumb
    out = os.path.join(folder, "_thumbs"); os.makedirs(out, exist_ok=True); n = 0
    stems = {f.rsplit(".", 1)[0] for f in os.listdir(folder) if f.endswith(".png")}
    for stem in sorted(stems):
        make_thumb(os.path.join(folder, stem + ".png"), os.path.join(out, f"{stem}.w{width}.jpg"), width); n += 1
    return n


def check() -> int:
    man = json.load(open(os.path.join(APP, "pipelines", "MANIFEST.json")))
    bad = 0
    for name, files in man["pipelines"].items():
        for rec in files:
            vend = os.path.join(APP, "pipelines", name, rec["file"]); src = os.path.join(REPO, rec["source"])
            if not os.path.exists(vend) or sha(vend) != rec["sha256"]:
                print(f"MODIFIED IN APP: pipelines/{name}/{rec['file']}"); bad += 1
            elif os.path.exists(src) and sha(src) != rec["sha256"]:
                print(f"REPO CHANGED SINCE SYNC: {rec['source']}  (run sync_pipelines.py)"); bad += 1
    print("pipelines in sync with the repository" if not bad else f"{bad} file(s) out of sync")
    return 1 if bad else 0


def _write_set(key: str, files: dict, man: dict, previews: bool = True) -> None:
    dst = os.path.join(APP, "reference_results", key); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for fname, src in files.items(): shutil.copy2(src, os.path.join(dst, fname))
    n_prev = pdf_previews(dst) if previews else 0
    if previews: make_thumbs(dst)
    man["reference"][key] = [{"file": f, "source": os.path.relpath(s, REPO).replace(os.sep, "/"), "sha256": sha(s)} for f, s in files.items()]
    print(f"reference_results/{key}: {len(files)} files" + (f" (+{n_prev} PDF previews)" if n_prev else ""))


def sync() -> None:
    man = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "pipelines": {}, "reference": {}}
    for name, spec in PIPELINES.items():
        dst = os.path.join(APP, "pipelines", name); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
        files = collect(spec)
        if not files: sys.exit(f"no scripts found for {name}")
        for fname, src in files.items(): shutil.copy2(src, os.path.join(dst, fname))
        man["pipelines"][name] = [{"file": f, "source": os.path.relpath(s, REPO).replace(os.sep, "/"), "sha256": sha(s)} for f, s in files.items()]
        print(f"pipelines/{name}: {len(files)} scripts")
    seen = set()
    for key, spec in REFERENCE.items():
        files = collect(spec); seen |= {sha(s) for s in files.values()}
        _write_set(key, files, man, previews=not key.endswith("_docs"))
    for key, mapping in RENAMED.items():
        files = {t: os.path.join(REPO, src) for t, src in mapping.items() if os.path.exists(os.path.join(REPO, src))}
        seen |= {sha(s) for s in files.values()}
        _write_set(key, files, man, previews=False)
    arch, arch_docs = collect_archive(seen)
    arch, dups = drop_current_copies(arch)
    man["duplicates"] = dups
    print(f"earlier-version figures identical to a current figure (not repeated): {len({d['same_as'] for d in dups})}")
    _write_set("E1_archive", arch, man); _write_set("archive_docs", arch_docs, man, previews=False)
    json.dump(man, open(os.path.join(APP, "pipelines", "MANIFEST.json"), "w"), indent=1)
    print("wrote pipelines/MANIFEST.json")


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else (sync() or 0))
