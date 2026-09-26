"""
run_all_P2.py  --  run the whole Paradigm 2 pipeline in the right order

The scripts (like Paradigm 1's) expect their inputs in the SAME folder. This runner copies every
script from the Code/ sub-folders plus pooled_data_P2.csv into a fresh flat work folder
(Code/_run/), runs the full sequence there, then files each output back where it belongs
(CSV tables next to their script's sub-folder, figures under ../Figures/<sub-folder>/).

    python run_all_P2.py                    # everything (Bayesian steps need PyMC -- see RUN_GUIDE_P2.md)
    python run_all_P2.py --skip-bayes-fits  # reuse the saved Bayesian fit tables; re-plot everything else
    python run_all_P2.py --data "C:/path/to/CIR files"   # rebuild pooled_data_P2.csv from the raw CIR files first

Stops at the first failing step and prints its log tail.
"""
import os, sys, glob, shutil, subprocess, time
CODE = os.path.dirname(os.path.abspath(__file__)); FIGS = os.path.join(os.path.dirname(CODE), "Figures")
SUBS = ["DDM", "Bayesian", "NDT", "SRT Analysis", "Vincentile", "Supplementary", "Validation"]

STEPS = ["DDM_fit.py", "DDM_figures.py", "DDM_conceptual.py", "NDT_barchart.py", "vincentile_figures.py",
         "why_saccadic_t0_floors.py", "SRT_identifiability_check.py", "SRT_fixed_t0_analysis.py",
         "Bayesian_HRT_fit.py", "Bayesian_SRT_fit.py", "Bayesian_SRT_ndt.py",
         "Bayesian_figures.py", "Bayesian_conceptual.py", "NDT_barchart_bayesian.py",
         "HRT_floor_control.py", "SRT_QA_flag_sensitivity.py", "direction_check.py",
         "dissociation_tests.py", "parameter_recovery_P2.py", "make_breakdown.py"]
BAYES_FITS = {"Bayesian_HRT_fit.py", "Bayesian_SRT_fit.py", "Bayesian_SRT_ndt.py"}

# where each output goes: filename prefix -> sub-folder (first match wins)
DEST = [("DDM_", "DDM"), ("ddm_", "DDM"), ("Bayesian_", "Bayesian"), ("bayes_", "Bayesian"), ("NDT_", "NDT"),
        ("SRT_identifiability", "SRT Analysis"), ("SRT_fixedt0", "SRT Analysis"), ("why_saccadic", "SRT Analysis"),
        ("SRT_QA_flag", "Supplementary"), ("vincentile_", "Vincentile"), ("HRT_floor_control", "Supplementary"),
        ("direction_check", "Supplementary"), ("dissociation_tests", "Supplementary"), ("parameter_recovery", "Supplementary"),
        ("P1_parity", "Validation")]

def dest(fname):
    for pre, sub in DEST:
        if fname.startswith(pre): return sub
    return None

def distribute(run_dir, code_dir=CODE, fig_dir=FIGS):
    """File outputs from a flat run folder into Code/<sub>/ (tables) and Figures/<sub>/ (figures)."""
    moved = 0
    for f in sorted(os.listdir(run_dir)):
        sub = dest(f); ext = os.path.splitext(f)[1].lower()
        if f == "P2_Technical_Breakdown.md":
            os.makedirs(os.path.join(os.path.dirname(code_dir), "Documents"), exist_ok=True)
            shutil.copy2(os.path.join(run_dir, f), os.path.join(os.path.dirname(code_dir), "Documents", f)); moved += 1; continue
        if sub is None or ext not in (".csv", ".pdf", ".png"): continue
        out = os.path.join(fig_dir if ext in (".pdf", ".png") else code_dir, sub)
        os.makedirs(out, exist_ok=True); shutil.copy2(os.path.join(run_dir, f), os.path.join(out, f)); moved += 1
    return moved

def main():
    skip = "--skip-bayes-fits" in sys.argv
    run = os.path.join(CODE, "_run"); shutil.rmtree(run, ignore_errors=True); os.makedirs(run)
    for sub in SUBS:
        for f in glob.glob(os.path.join(CODE, sub, "*.py")): shutil.copy2(f, run)
    for f in ["build_pooled_data_P2.py", "pooled_data_P2.csv"]:
        if os.path.exists(os.path.join(CODE, f)): shutil.copy2(os.path.join(CODE, f), run)
    if "--data" in sys.argv:
        src = sys.argv[sys.argv.index("--data") + 1]
        subprocess.run([sys.executable, "build_pooled_data_P2.py", src], cwd=run, check=True)
    if not os.path.exists(os.path.join(run, "pooled_data_P2.csv")):
        sys.exit("pooled_data_P2.csv missing -- pass --data <folder with the CIR*_TRIAL_Summary files>")
    pr = os.path.join(CODE, "Validation", "P1_parity_results.csv")
    if os.path.exists(pr): shutil.copy2(pr, run)   # the breakdown reports the saved Paradigm 1 parity result
    if skip:   # reuse the saved Bayesian tables so the figure/supplementary steps can run without PyMC
        for f in glob.glob(os.path.join(CODE, "Bayesian", "Bayesian_*.csv")): shutil.copy2(f, run)
    for s in STEPS:
        if skip and s in BAYES_FITS: print(f"-- skip {s} (using saved tables)"); continue
        t = time.time(); print(f">> {s}", flush=True)
        r = subprocess.run([sys.executable, s], cwd=run, capture_output=True, text=True)
        open(os.path.join(run, f"log_{s[:-3]}.txt"), "w").write(r.stdout + r.stderr)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-3000:]); sys.exit(f"FAILED at {s} (see Code/_run/log_{s[:-3]}.txt)")
        print(f"   ok ({time.time() - t:.0f}s)", flush=True)
    print(f"filed {distribute(run)} outputs into Code/<sub>/ and Figures/<sub>/  (work folder: Code/_run/)")

if __name__ == "__main__":
    main()
