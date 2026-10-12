"""Regularization study on Task A (3-class subtype) — how each regulariser controls over-fitting.

Figures
  reg_logreg_paths.png       L2 vs L1 vs elastic-net: train/CV macro-F1 and #non-zero genes vs C
  reg_learning_curves.png    (almost) unregularised vs tuned elastic-net logistic regression
  reg_tree_pruning.png       decision tree: max_depth and cost-complexity pruning (ccp_alpha) paths
  reg_svm_c.png              RBF-SVM: margin softness C
  reg_random_forest.png      random forest: tree depth and minimum leaf size (variance control)
Table
  reg_summary.csv            train vs CV score and generalisation gap for each setting
"""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, validation_curve

from glioma.build_dataset import TASK_A
from glioma.config import SEED, TABLES
from glioma.data import load_task
from glioma.models import make_pipeline
from glioma.plots import INK2, SERIES, save
from glioma.train import MODELS_DIR, cv5

warnings.filterwarnings("ignore")
ROWS: list[dict] = []


def _curve(ax, x, tr, va, label, color, logx=True):
    ax.plot(x, va.mean(1), color=color, label=f"{label} — CV")
    ax.fill_between(x, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=color, alpha=0.15, lw=0)
    ax.plot(x, tr.mean(1), color=color, ls="--", lw=1.4, label=f"{label} — train")
    if logx:
        ax.set_xscale("log")


def _log(setting, param, value, tr, va):
    i = int(np.argmax(va.mean(1)))
    ROWS.append({"setting": setting, "param": param, "best_value": value[i],
                 "train_f1": tr.mean(1)[i], "cv_f1": va.mean(1)[i], "gap": tr.mean(1)[i] - va.mean(1)[i],
                 "cv_f1_at_weakest_reg": va.mean(1)[-1] if param in ("C", "max_depth") else va.mean(1)[0],
                 "gap_at_weakest_reg": (tr.mean(1) - va.mean(1))[-1] if param in ("C", "max_depth")
                 else (tr.mean(1) - va.mean(1))[0]})


def logreg_paths(X, y):
    Cs = np.logspace(-3, 1.5, 10)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
    for (name, l1r, solver), colr in zip([("L2 (ridge)", 0.0, "lbfgs"), ("Elastic-net (0.5)", 0.5, "saga"),
                                          ("L1 (lasso)", 1.0, "saga")], SERIES):
        pipe = make_pipeline(TASK_A, "logreg", model=LogisticRegression(
            solver=solver, l1_ratio=l1r, max_iter=3000, tol=1e-3, class_weight="balanced", random_state=SEED))
        pipe.set_params(feat__k_genes=2000)
        tr, va = validation_curve(pipe, X, y, param_name="model__C", param_range=Cs, cv=cv5(),
                                  scoring="f1_macro", n_jobs=8)
        _curve(axes[0], Cs, tr, va, name, colr)
        _log(f"logreg {name}", "C", Cs, tr, va)
        nz = []
        for C in Cs:
            p = clone(pipe).set_params(model__C=C).fit(X, y)
            nz.append(int((np.abs(p.named_steps["model"].coef_).max(0) > 1e-6).sum()))
        axes[1].plot(Cs, nz, color=colr, marker="o", ms=4, label=name)
    axes[0].set_xlabel("C  (← stronger regularisation   weaker →)")
    axes[0].set_ylabel("macro-F1")
    axes[0].set_title("Logistic regression: penalty strength vs fit")
    axes[0].legend(fontsize=7.5, ncol=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("features with non-zero weight")
    axes[1].set_title("Sparsity: L1 selects genes, L2 keeps all 2,095")
    axes[1].legend()
    save(fig, "reg_logreg_paths")


def learning_curves(X, y, tuned_lr):
    sizes = np.linspace(0.15, 1.0, 7)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    weak = make_pipeline(TASK_A, "logreg", model=LogisticRegression(
        C=1e4, max_iter=5000, class_weight="balanced", random_state=SEED)).set_params(feat__k_genes=2000)
    for pipe, label, colr in [(weak, "C=10⁴ (≈ unregularised)", SERIES[1]), (tuned_lr, "tuned elastic-net", SERIES[0])]:
        n, tr, va = learning_curve(pipe, X, y, train_sizes=sizes, cv=cv5(), scoring="f1_macro", n_jobs=8,
                                   shuffle=True, random_state=SEED)
        _curve(ax, n, tr, va, label, colr, logx=False)
        ROWS.append({"setting": f"learning curve: {label}", "param": "n_train", "best_value": int(n[-1]),
                     "train_f1": tr.mean(1)[-1], "cv_f1": va.mean(1)[-1], "gap": tr.mean(1)[-1] - va.mean(1)[-1]})
    ax.set_xlabel("training samples")
    ax.set_ylabel("macro-F1")
    ax.set_title("Learning curves: regularisation closes the train–CV gap")
    ax.legend(fontsize=7.5)
    save(fig, "reg_learning_curves")


def tree_pruning(X, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    pipe = make_pipeline(TASK_A, "tree").set_params(feat__k_genes=2000)
    depths = np.arange(1, 16)
    tr, va = validation_curve(pipe, X, y, param_name="model__max_depth", param_range=depths, cv=cv5(),
                              scoring="f1_macro", n_jobs=8)
    _curve(axes[0], depths, tr, va, "depth", SERIES[0], logx=False)
    _log("tree max_depth", "max_depth", depths, tr, va)
    axes[0].set_xlabel("max_depth")
    axes[0].set_title("Decision tree: depth limit")
    alphas = np.logspace(-4, -0.7, 12)
    tr, va = validation_curve(pipe, X, y, param_name="model__ccp_alpha", param_range=alphas, cv=cv5(),
                              scoring="f1_macro", n_jobs=8)
    _curve(axes[1], alphas, tr, va, "ccp_alpha", SERIES[1])
    _log("tree ccp_alpha", "ccp_alpha", alphas, tr, va)
    axes[1].set_xlabel("ccp_alpha  (weaker ← pruning → stronger)")
    axes[1].set_title("Decision tree: cost-complexity pruning")
    for a in axes:
        a.set_ylabel("macro-F1")
        a.legend(fontsize=7.5)
    save(fig, "reg_tree_pruning")


def svm_c(X, y):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    pipe = make_pipeline(TASK_A, "svm_rbf").set_params(feat__k_genes=2000)
    Cs = np.logspace(-2, 3, 11)
    tr, va = validation_curve(pipe, X, y, param_name="model__estimator__C", param_range=Cs, cv=cv5(),
                              scoring="f1_macro", n_jobs=8)
    _curve(ax, Cs, tr, va, "RBF-SVM", SERIES[0])
    _log("svm_rbf C", "C", Cs, tr, va)
    ax.set_xlabel("C  (← wider, softer margin   harder margin →)")
    ax.set_ylabel("macro-F1")
    ax.set_title("SVM: margin softness C")
    ax.legend(fontsize=7.5)
    save(fig, "reg_svm_c")


def random_forest_capacity(X, y):
    """Random forest: bagging already averages out variance; depth and leaf size limit each tree further."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    pipe = make_pipeline(TASK_A, "random_forest").set_params(feat__k_genes=2000, model__n_estimators=300)
    depths = np.array([2, 3, 4, 6, 8, 12, 16, 24])
    tr, va = validation_curve(pipe, X, y, param_name="model__max_depth", param_range=depths, cv=cv5(),
                              scoring="f1_macro", n_jobs=8)
    _curve(axes[0], depths, tr, va, "max_depth", SERIES[0], logx=False)
    _log("random_forest max_depth", "max_depth", depths, tr, va)
    axes[0].set_xlabel("max_depth of each tree")
    axes[0].set_title("Random forest: tree depth")
    leaves = np.array([1, 2, 4, 8, 16, 32, 64])
    tr, va = validation_curve(pipe, X, y, param_name="model__min_samples_leaf", param_range=leaves, cv=cv5(),
                              scoring="f1_macro", n_jobs=8)
    _curve(axes[1], leaves, tr, va, "min_samples_leaf", SERIES[1])
    _log("random_forest min_samples_leaf", "min_samples_leaf", leaves, tr, va)
    axes[1].set_xlabel("min_samples_leaf  (weaker ← regularisation → stronger)")
    axes[1].set_title("Random forest: minimum leaf size")
    for a in axes:
        a.set_ylabel("macro-F1")
        a.legend(fontsize=7.5)
    save(fig, "reg_random_forest")


def main() -> None:
    d = load_task(TASK_A)
    X, y = d.X_train, d.y_train.map({c: i for i, c in enumerate(d.classes)}).astype(int)
    best = json.loads((MODELS_DIR / TASK_A / "best_params.json").read_text())
    tuned_lr = make_pipeline(TASK_A, "logreg").set_params(**best["logreg"])
    logreg_paths(X, y)
    learning_curves(X, y, tuned_lr)
    tree_pruning(X, y)
    svm_c(X, y)
    random_forest_capacity(X, y)
    pd.DataFrame(ROWS).round(4).to_csv(TABLES / "reg_summary.csv", index=False)
    print(pd.DataFrame(ROWS).round(3).to_string(), flush=True)
    print(INK2 and "done")


if __name__ == "__main__":
    main()
