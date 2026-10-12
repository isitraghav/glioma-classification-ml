"""Run the whole project end to end (or any slice of it) with one command.

  uv run glioma-all                          # every stage, in order
  uv run glioma-all --from evaluate          # re-run only the cheap reporting stages
  uv run glioma-all --only train --fast      # smoke-test training
  uv run glioma-all --list                   # show stages

Each stage runs in its own process (`python -m glioma.<module>`), exactly as its own CLI would.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [  # (name, module, what it does, fits models?)
    ("download", "download", "fetch raw data (aria2c + GDC API), write MANIFEST.csv", False),
    ("literature", "literature", "scrape PubMed, extract published baselines", False),
    ("build", "build_dataset", "merge sources into analysis-ready parquet files", False),
    ("eda", "eda", "exploratory data analysis figures and tables", False),
    ("train", "train", "ablation, default CV, tuning, ensembles, repeated + nested CV, refit", True),
    ("regularization", "regularization", "regularisation study figures (Task A)", True),
    ("tuning-plots", "tuning_analysis", "hyperparameter-impact figures", False),
    ("evaluate", "evaluate", "held-out test metrics, CIs, statistical tests, SHAP", False),
    ("external", "external", "external validation on GEO cohorts", True),
    ("report", "report", "assemble reports/report.md and models/MODEL_CARD.md", False),
]
NAMES = [s[0] for s in STAGES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0])
    ap.add_argument("--to", dest="end", choices=NAMES, default=NAMES[-1])
    ap.add_argument("--only", choices=NAMES)
    ap.add_argument("--fast", action="store_true", help="tiny tuning budgets for the train stage")
    ap.add_argument("--tasks", nargs="+", help="restrict train/evaluate/tuning-plots to these tasks")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for name, _, desc, fits in STAGES:
            print(f"{name:15s} {'[fits models] ' if fits else ''}{desc}")
        return
    lo, hi = (NAMES.index(a.only),) * 2 if a.only else (NAMES.index(a.start), NAMES.index(a.end))
    for name, module, desc, _ in STAGES[lo:hi + 1]:
        cmd = [sys.executable, "-m", f"glioma.{module}"]
        if name == "train" and a.fast:
            cmd.append("--fast")
        if a.tasks and name in ("train", "evaluate", "tuning-plots"):
            cmd += ["--tasks", *a.tasks]
        print(f"\n▶ {name}: {desc}", flush=True)
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit(f"stage '{name}' failed — fix it and resume with `glioma-all --from {name}`")
        print(f"✓ {name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
