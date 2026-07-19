#!/usr/bin/env python
"""
Command-line runner for the whole pipeline (no GUI).

    python run_pipeline.py                         # example data, default settings
    python run_pipeline.py --data pooled_data.csv  # your data
    python run_pipeline.py --config my_run.yaml    # everything from a config file
    python run_pipeline.py --preview               # fast MLE preview instead of NUTS

Writes repo-compatible CSVs, figures, and an HTML report to the output folder.
"""

import argparse

from kinarm_rt.pipeline import run_pipeline, load_config


def main():
    ap = argparse.ArgumentParser(description="KINARM RT analysis pipeline")
    ap.add_argument("--config", help="YAML/JSON config file")
    ap.add_argument("--data", help="path to a trial CSV (overrides config)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--preview", action="store_true", help="fast MLE preview (no NUTS)")
    ap.add_argument("--draws", type=int, help="posterior draws (Bayesian)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg["data"] = args.data
    if args.out:
        cfg["out"] = args.out
    if args.preview:
        cfg["mode"] = "preview"
    if args.draws:
        cfg["sampler"]["draws"] = args.draws
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
