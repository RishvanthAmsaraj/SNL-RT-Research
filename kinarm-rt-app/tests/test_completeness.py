"""Everything the repository holds as results is in the app: every figure file, every table and every document, by content."""
import hashlib, json, os, re
import pytest
from conftest import APP, REPO

FIGURE_DIRS = ["Current Pipeline/Figures", "Paradigm 2 Pipeline/Figures", "Deprecated Pipelines", "Working Iterations"]
TABLE_DIRS = ["Current Pipeline/Code", "Paradigm 2 Pipeline/Code", "Deprecated Pipelines", "Working Iterations"]
DOC_DIRS = ["Current Pipeline/Documents", "Paradigm 2 Pipeline/Documents", "Deprecated Pipelines", "Working Iterations"]
SKIP_TABLES = ("pooled_data", "example_ddm_data")
# Per-participant tables (de-identified CMT001.../CIR001... rows) are intentionally not
# shipped, so the completeness check must not require them to be present in the app.
PARTICIPANT_ID = re.compile(r"\b(CMT[0-9]{3,4}|CIR[0-9]{3})\b")


def _is_participant_table(p):
    if not p.lower().endswith(".csv"):
        return False
    try:
        with open(p, "r", errors="ignore") as f:
            return bool(PARTICIPANT_ID.search(f.read(65536)))
    except OSError:
        return False


def _sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def _shipped():
    """Files shipped in the app, plus earlier-version copies of current figures, which are recorded (same_as) rather than repeated."""
    man = json.load(open(os.path.join(APP, "pipelines", "MANIFEST.json")))
    return {rec["sha256"] for recs in man["reference"].values() for rec in recs} | {d["sha256"] for d in man.get("duplicates", [])}


def _files(dirs, exts, skip=()):
    for d in dirs:
        for root, _, fs in os.walk(os.path.join(REPO, d)):
            for f in fs:
                if not f.lower().endswith(exts) or f.startswith(skip):
                    continue
                p = os.path.join(root, f)
                if _is_participant_table(p):
                    continue
                yield p


needs_repo = pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, "Current Pipeline")), reason="app copied out of the repository")


@needs_repo
def test_every_figure_file_in_the_repository_is_in_the_app():
    shipped = _shipped()
    files = list(_files(FIGURE_DIRS, (".png", ".pdf"))); assert len(files) >= 200
    missing = [os.path.relpath(p, REPO) for p in files if _sha(p) not in shipped]
    assert not missing, missing[:20]


@needs_repo
def test_every_table_in_the_repository_is_in_the_app():
    shipped = _shipped()
    files = list(_files(TABLE_DIRS, (".csv",), SKIP_TABLES)); assert len(files) >= 15
    missing = [os.path.relpath(p, REPO) for p in files if _sha(p) not in shipped]
    assert not missing, missing[:20]


@needs_repo
def test_every_document_in_the_repository_is_in_the_app():
    shipped = _shipped()
    files = list(_files(DOC_DIRS, (".md",))); assert len(files) >= 30
    missing = [os.path.relpath(p, REPO) for p in files if _sha(p) not in shipped]
    assert not missing, missing[:20]


def test_shipped_files_match_the_manifest():
    man = json.load(open(os.path.join(APP, "pipelines", "MANIFEST.json")))
    for key, recs in man["reference"].items():
        for rec in recs:
            p = os.path.join(APP, "reference_results", key, rec["file"])
            assert os.path.exists(p) and _sha(p) == rec["sha256"], (key, rec["file"])


def test_deprecated_holds_only_older_figures():
    """Earlier-version copies that are identical to a current figure are not shown as deprecated; corrected ones are."""
    man = json.load(open(os.path.join(APP, "pipelines", "MANIFEST.json")))
    shown = {f.rsplit(".", 1)[0] for f in os.listdir(os.path.join(APP, "reference_results", "E1_archive")) if f.endswith((".png", ".pdf"))}
    for d in man.get("duplicates", []):
        tag = d["source"].split("/")[1] if d["source"].startswith("Deprecated Pipelines") else "wi"
        assert not any(st.endswith("__" + d["same_as"]) and st.startswith(("v3__",) if "Ver 3" in d["source"] else ("v2_5__",) if "Ver 2.5" in d["source"] else ("wi__",)) for st in shown), d
    for corrected in ("DDM_summary", "Bayesian_summary", "Bayesian_srt_ndt", "NDT_barchart_bayesian"):
        assert f"v3__{corrected}" in shown, corrected            # the pre-correction version is kept to show the mistake


def test_insights_summary_is_shipped_and_aggregate():
    """The numbers the explanations show come from a pre-computed summary, not from the
    per-participant tables (which are not shipped); the summary must be aggregate only."""
    p = os.path.join(APP, "pipelines", "insights_summary.json")
    assert os.path.exists(p), "pipelines/insights_summary.json missing"
    text = open(p).read()
    assert not PARTICIPANT_ID.search(text), "participant identifiers in the insights summary"
    s = json.load(open(p))
    assert set(s) == {"E1", "E2"} and s["E1"]["headline"].get("hand_t0") and s["E2"]["headline"].get("hand_t0")
