"""Visualise how each hyperparameter moved the CV score (from the saved search histories)."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

from glioma.build_dataset import TASK_A, TASK_B, TASK_C
from glioma.config import TABLES
from glioma.plots import INK2, MUTED, SERIES, save

PANELS = [  # (model, param column, label, log-x)
    ("logreg", "param_model__C", "Logistic reg.: C", True),
    ("logreg", "param_model__l1_ratio", "Logistic reg.: l1_ratio (0=L2, 1=L1)", False),
    ("svm_rbf", "param_model__estimator__C", "RBF SVM: C", True),
    ("svm_rbf", "param_model__estimator__gamma", "RBF SVM: gamma", True),
    ("knn", "param_model__n_neighbors", "k-NN: n_neighbors", False),
    ("tree", "param_model__max_depth", "Decision tree: max_depth", False),
    ("tree", "param_model__ccp_alpha", "Decision tree: ccp_alpha", True),
    ("random_forest", "param_model__min_samples_leaf", "Random forest: min_samples_leaf", False),
]


def run(task: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    axes = axes.ravel()
    i = 0
    for model, col, label, logx in PANELS:
        f = TABLES / f"{task}_search_{model}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        if col not in s:
            continue
        x = pd.to_numeric(s[col], errors="coerce")
        ax = axes[i]
        ax.scatter(x, s["mean_test_score"], s=22, color=SERIES[0], edgecolor="white", lw=0.4, label="CV")
        if "mean_train_score" in s:
            ax.scatter(x, s["mean_train_score"], s=14, color=SERIES[1], marker="^", alpha=0.7, label="train")
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label, fontsize=9.5)
        ax.set_ylabel("macro-F1", fontsize=8)
        i += 1
    axes[0].legend(fontsize=7)

    # number of genes (feature-step hyperparameter) across all random searches
    ks = []
    for model in ["logreg", "svm_linear", "svm_rbf", "knn", "tree", "random_forest"]:
        f = TABLES / f"{task}_search_{model}.csv"
        if f.exists():
            s = pd.read_csv(f)
            if "param_feat__k_genes" in s:
                ks.append(s[["param_feat__k_genes", "mean_test_score"]].assign(model=model))
    if ks:
        k = pd.concat(ks)
        k["rel"] = k["mean_test_score"] - k.groupby("model")["mean_test_score"].transform("mean")
        ax = axes[i]
        grp = k.groupby("param_feat__k_genes")["rel"]
        ax.bar(range(len(grp)), grp.mean(), color=SERIES[0], width=0.6, yerr=grp.std(),
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        ax.set_xticks(range(len(grp)), [str(int(v)) for v in grp.mean().index])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title("Top-k genes (all models)", fontsize=9.5)
        ax.set_ylabel("F1 vs model mean", fontsize=8)
        i += 1

    for j in range(i, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{task}: effect of individual hyperparameters (each dot = one random-search candidate)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, f"tune_{task}_hyperparameters")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[TASK_A, TASK_C, TASK_B])
    for t in ap.parse_args().tasks:
        if (TABLES / f"{t}_tuning_best.csv").exists():
            run(t)


if __name__ == "__main__":
    main()
