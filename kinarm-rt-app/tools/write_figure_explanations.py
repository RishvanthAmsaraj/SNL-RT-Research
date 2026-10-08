"""write_figure_explanations.py -- write FIGURE_EXPLANATIONS.md from the app's explanation catalogue (kinarm_rt/explain.py),
so the same words the app shows next to each figure can be read, printed and reviewed as one document.
    python tools/write_figure_explanations.py"""
import os, sys
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, APP)
from kinarm_rt import explain as X

def section(f):
    title = f.title + (" (one per target speed)" if f.per_speed else "")
    lines = [f"### {title}", "", f"*Group: {X.CATEGORY_LABEL[f.category]}. File: `{f.stem.replace('{s}', '<speed>')}`.*", "",
             f"**What it shows.** {f.what}", ""]
    if f.why: lines += [f"**Why it is here.** {f.why}", ""]
    lines += [f"**How to read it.** {f.read}", "", f"**What it means for the project.** {f.meaning}", ""]
    if f.caveat: lines += [f"**Keep in mind.** {f.caveat}", ""]
    return lines

out = ["# Figure explanations", "",
       "The explanations the KINARM RT app shows next to each figure, collected in one place. In the app, each figure page "
       "also lists the numbers behind it for the results being shown (\"In these results\"), computed from the tables.", ""]
for cat in ("main", "diagnostic", "method_a", "deprecated"):
    figs = [f for f in X.FIGURES if f.category == cat]
    if figs:
        out += [f"## {X.CATEGORY_LABEL[cat]}", ""]
        for f in figs: out += section(f)
out += ["## Experiment comparison", ""]
for f in X.COMPARE_FIGURES: out += section(f)
out += ["## Earlier pipeline versions", "",
        "Figures from pipeline versions 2, 2.5 and 3 and the working iterations appear under Deprecated with the version in their "
        "title. Only genuinely older figures are shown: copies identical to a current figure are recorded in pipelines/MANIFEST.json "
        "(\"duplicates\") rather than repeated, and the pre-correction versions of the figures fixed in October 2026 are kept because "
        "they show the mistake that was corrected.", ""]
open(os.path.join(APP, "FIGURE_EXPLANATIONS.md"), "w", encoding="utf-8").write("\n".join(out))
print("wrote FIGURE_EXPLANATIONS.md")
