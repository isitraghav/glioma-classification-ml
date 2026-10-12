"""Performance analysis on the untouched 20 % test split + statistical model comparison.

Per task
  <task>_test_metrics.csv         every model x {acc, balanced acc, macro-F1, MCC, kappa, AUC, PR-AUC, log-loss}
  <task>_test_bootstrap_ci.csv    95 % bootstrap CIs (1,000 resamples) for the key metrics
  <task>_cv_vs_test.csv           tuned repeated-CV mean +- sd next to test score, plus default-CV score
  <task>_stat_tests.csv           corrected resampled t-test (Nadeau-Bengio) on CV folds + exact McNemar on test
  <task>_per_class.csv            precision / recall / F1 per class for the best model
  <task>_errors.csv               misclassified test patients with clinical context
  <task>_shap_top.csv / <task>_logreg_top.csv   interpretation
Figures: eval_<task>_*.png
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar

from glioma.build_dataset import TASK_A, TASK_B, TASK_C
from glioma.config import SEED, TABLES
from glioma.data import load_task, meta
from glioma.models import CLASSICAL, ENSEMBLES
from glioma.plots import INK2, MUTED, SEQ, SERIES, colors_for, save
from glioma.train import MODELS_DIR

warnings.filterwarnings("ignore")
ORDER = CLASSICAL + ENSEMBLES
PRETTY = {"logreg": "Logistic reg. (EN)", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "knn": "k-NN",
          "tree": "Decision tree", "random_forest": "Random forest", "soft_voting": "Soft voting"}


def metrics(y, proba, k) -> dict:
    pred = proba.argmax(1)
    out = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
           "f1_macro": f1_score(y, pred, average="macro"), "f1_weighted": f1_score(y, pred, average="weighted"),
           "mcc": matthews_corrcoef(y, pred), "kappa": cohen_kappa_score(y, pred),
           "log_loss": log_loss(y, np.clip(proba, 1e-7, 1), labels=list(range(k)))}
    if k == 2:
        out["roc_auc"] = roc_auc_score(y, proba[:, 1])
        out["pr_auc"] = average_precision_score(y, proba[:, 1])
    else:
        Y = label_binarize(y, classes=list(range(k)))
        out["roc_auc"] = roc_auc_score(Y, proba, average="macro", multi_class="ovr")
        out["pr_auc"] = average_precision_score(Y, proba, average="macro")
    return out


def bootstrap(y, proba, k, n=1000) -> dict:
    rng = np.random.default_rng(SEED)
    vals = {"f1_macro": [], "balanced_accuracy": [], "roc_auc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < k:
            continue
        m = metrics(y[i], proba[i], k)
        for key in vals:
            vals[key].append(m[key])
    return {f"{key}_ci": (np.percentile(v, 2.5), np.percentile(v, 97.5)) for key, v in vals.items()}


def corrected_ttest(a, b, n_train, n_test) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for CV fold scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def evaluate_task(task: str) -> pd.DataFrame:
    d = load_task(task)
    k = len(d.classes)
    enc = {c: i for i, c in enumerate(d.classes)}
    y = d.y_test.map(enc).to_numpy()
    names = [n for n in ORDER if (MODELS_DIR / task / f"{n}.joblib").exists()]
    models = {n: joblib.load(MODELS_DIR / task / f"{n}.joblib") for n in names}
    probas = {n: m.predict_proba(d.X_test) for n, m in models.items()}

    rows, ci_rows = [], []
    for n in names:
        rows.append({"model": n, **metrics(y, probas[n], k)})
        ci = bootstrap(y, probas[n], k)
        ci_rows.append({"model": n, **{f"{m}_lo": v[0] for m, v in ci.items()},
                        **{f"{m}_hi": v[1] for m, v in ci.items()}})
    test = pd.DataFrame(rows).set_index("model")
    test.round(4).to_csv(TABLES / f"{task}_test_metrics.csv")
    ci = pd.DataFrame(ci_rows).set_index("model")
    ci.round(4).to_csv(TABLES / f"{task}_test_bootstrap_ci.csv")

    folds = pd.read_csv(TABLES / f"{task}_cv_tuned_folds.csv")
    default = pd.read_csv(TABLES / f"{task}_cv_default.csv").set_index("model")
    cvs = folds.groupby("model").agg(cv_f1=("test_f1_macro", "mean"), cv_f1_sd=("test_f1_macro", "std"),
                                     train_f1=("train_f1_macro", "mean"))
    comp = cvs.join(default["test_f1_macro"].rename("default_cv_f1")).join(test["f1_macro"].rename("test_f1"))
    comp = comp.join(ci[["f1_macro_ci_lo", "f1_macro_ci_hi"]]).reindex(names)
    comp["tuning_gain"] = comp["cv_f1"] - comp["default_cv_f1"]
    comp["overfit_gap"] = comp["train_f1"] - comp["cv_f1"]
    comp.round(4).to_csv(TABLES / f"{task}_cv_vs_test.csv")

    # best model is chosen on CV (never on test) — then its test score is reported
    best = comp["cv_f1"].idxmax()
    n_tr = len(d.y_train)
    st_rows = []
    pred_best = probas[best].argmax(1)
    for n in names:
        if n == best:
            continue
        a = folds.query("model == @best")["test_f1_macro"].to_numpy()
        b = folds.query("model == @n")["test_f1_macro"].to_numpy()
        t, p = corrected_ttest(a, b, n_tr * 0.8, n_tr * 0.2)
        pn = probas[n].argmax(1)
        tbl = [[np.sum((pred_best == y) & (pn == y)), np.sum((pred_best == y) & (pn != y))],
               [np.sum((pred_best != y) & (pn == y)), np.sum((pred_best != y) & (pn != y))]]
        mc = mcnemar(tbl, exact=True)
        st_rows.append({"best_model": best, "vs_model": n, "cv_f1_diff": a.mean() - b.mean(),
                        "corrected_t": t, "corrected_t_p": p, "mcnemar_b": tbl[0][1], "mcnemar_c": tbl[1][0],
                        "mcnemar_p": mc.pvalue})
    pd.DataFrame(st_rows).round(4).to_csv(TABLES / f"{task}_stat_tests.csv", index=False)

    rep = classification_report(y, pred_best, target_names=d.classes, output_dict=True)
    pd.DataFrame(rep).T.round(4).to_csv(TABLES / f"{task}_per_class.csv")

    # error analysis
    m = meta().reindex(d.X_test.index)
    err = pd.DataFrame({"true": d.y_test, "pred": [d.classes[i] for i in pred_best],
                        "p_max": probas[best].max(1)}, index=d.X_test.index)
    cols = [c for c in ["histology", "who_grade", "age", "purity", "batch_tss", "idh_status", "codel_1p19q",
                        "subtype_7"] if c in m]
    err = err.join(m[cols])
    err[err["true"] != err["pred"]].to_csv(TABLES / f"{task}_errors.csv")
    (TABLES / f"{task}_best_model.json").write_text(json.dumps({
        "best_by_cv": best, "cv_f1": comp.loc[best, "cv_f1"], "test_f1": comp.loc[best, "test_f1"],
        "test_ci": [comp.loc[best, "f1_macro_ci_lo"], comp.loc[best, "f1_macro_ci_hi"]],
        "n_test": int(len(y)), "n_errors": int((err["true"] != err["pred"]).sum())}, indent=1, default=float))

    plot_comparison(task, comp)
    plot_confusion(task, d.classes, y, probas, best, comp)
    plot_roc_calibration(task, d.classes, y, probas, best)
    interpret(task, d, models, best)
    print(f"\n=== {task}: best by CV = {best}")
    print(comp.round(3).to_string())
    return comp


def plot_comparison(task, comp):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(comp))
    w = 0.27
    ax.bar(x - w, comp["default_cv_f1"], w, color="#c3c2b7", label="default hyperparameters (CV)")
    ax.bar(x, comp["cv_f1"], w, color=SERIES[0], label="tuned (repeated CV, ± sd)",
           yerr=comp["cv_f1_sd"], error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    lo = comp["test_f1"] - comp["f1_macro_ci_lo"]
    hi = comp["f1_macro_ci_hi"] - comp["test_f1"]
    ax.bar(x + w, comp["test_f1"], w, color=SERIES[1], label="tuned — held-out test (95% bootstrap CI)",
           yerr=[lo, hi], error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    ax.set_xticks(x, [PRETTY.get(n, n) for n in comp.index], rotation=35, ha="right")
    floor = max(0, np.nanmin(comp[["default_cv_f1", "cv_f1", "f1_macro_ci_lo"]].to_numpy()) - 0.05)
    ax.set_ylim(floor, 1.0)
    ax.set_ylabel("macro-F1")
    split = sum(n in CLASSICAL for n in comp.index) - 0.5  # boundary between single models and ensembles
    ax.axvline(split, color=MUTED, lw=1, ls=":")
    ax.text(split + 0.1, floor + 0.01, "ensembles →", color=MUTED, fontsize=8)
    ax.set_title(f"{task}: default vs tuned (CV) vs held-out test")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(axis="x", visible=False)
    save(fig, f"eval_{task}_model_comparison")


def plot_confusion(task, classes, y, probas, best, comp):
    others = [n for n in ["logreg", "svm_rbf", "tree", "random_forest", "soft_voting"] if n in probas and n != best]
    show = [best] + others[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(4.0 * len(show), 3.9))
    for ax, n in zip(np.atleast_1d(axes), show):
        cm = confusion_matrix(y, probas[n].argmax(1), labels=range(len(classes)))
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8 if len(classes) < 5 else 7,
                        color="white" if cmn[i, j] > 0.55 else INK2)
        short = [c.replace("IDHmut-", "mut-").replace("-like", "") for c in classes]
        ax.set_xticks(range(len(classes)), short, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(classes)), short, fontsize=7.5)
        ax.grid(False)
        ax.set_xlabel("predicted")
        tag = " (best by CV)" if n == best else ""
        ax.set_title(f"{PRETTY.get(n, n)}{tag}\ntest F1 = {comp.loc[n, 'test_f1']:.3f}", fontsize=9.5)
    np.atleast_1d(axes)[0].set_ylabel("true")
    save(fig, f"eval_{task}_confusion")


def plot_roc_calibration(task, classes, y, probas, best):
    k = len(classes)
    p = probas[best]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if k == 2:
        fpr, tpr, _ = roc_curve(y, p[:, 1])
        axes[0].plot(fpr, tpr, color=SERIES[0], label=f"{classes[1]} (AUC {roc_auc_score(y, p[:, 1]):.3f})")
    else:
        Y = label_binarize(y, classes=list(range(k)))
        cols = colors_for(classes)
        for i, c in enumerate(classes):
            if Y[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(Y[:, i], p[:, i])
            axes[0].plot(fpr, tpr, color=cols[c], lw=1.6, label=f"{c} (AUC {roc_auc_score(Y[:, i], p[:, i]):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
    axes[0].set_xlabel("false-positive rate")
    axes[0].set_ylabel("true-positive rate")
    axes[0].set_title(f"One-vs-rest ROC — {PRETTY.get(best, best)}")
    axes[0].legend(fontsize=7, loc="lower right")

    # top-label reliability diagram + expected calibration error for a few models
    bins = np.linspace(0, 1, 11)
    show = [n for n in [best, "logreg", "random_forest", "svm_rbf"] if n in probas]
    show = list(dict.fromkeys(show))[:3]
    ece_rows = {}
    for n, colr in zip(show, SERIES):
        conf = probas[n].max(1)
        corr = (probas[n].argmax(1) == y).astype(float)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs, ys, ece = [], [], 0.0
        for b in range(10):
            s = idx == b
            if s.sum():
                xs.append(conf[s].mean())
                ys.append(corr[s].mean())
                ece += s.mean() * abs(conf[s].mean() - corr[s].mean())
        ece_rows[n] = ece
        axes[1].plot(xs, ys, marker="o", ms=4, color=colr, label=f"{PRETTY.get(n, n)} (ECE {ece:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
    axes[1].set_xlabel("predicted confidence (top label)")
    axes[1].set_ylabel("observed accuracy")
    axes[1].set_title("Calibration (reliability diagram)")
    axes[1].legend(fontsize=7.5, loc="upper left")
    pd.Series(ece_rows, name="ECE").to_csv(TABLES / f"{task}_calibration_ece.csv")
    save(fig, f"eval_{task}_roc_calibration")


def interpret(task, d, models, best):
    """SHAP (TreeExplainer) on the tuned random forest + elastic-net coefficients."""
    import shap
    if "random_forest" in models:
        pipe = models["random_forest"]
        prep = pipe[:-1]
        Xt = prep.transform(d.X_test)
        names = list(pipe.named_steps["feat"].get_feature_names_out())
        if Xt.shape[1] != len(names):  # PCA step active -> name components
            names = [f"PC{i + 1}" for i in range(Xt.shape[1])]
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(np.asarray(Xt))
        sv = np.asarray(sv)
        # normalise to (n_samples, n_features, n_classes)
        if sv.ndim == 2:
            sv = sv[:, :, None]
        elif sv.shape[0] == len(d.classes) and sv.shape[1] == Xt.shape[0]:
            sv = np.transpose(sv, (1, 2, 0))
        imp = pd.DataFrame(np.abs(sv).mean(0), index=names,
                           columns=d.classes if sv.shape[2] == len(d.classes) else ["shap"])
        imp["total"] = imp.sum(1)
        imp = imp.sort_values("total", ascending=False)
        imp.head(30).round(5).to_csv(TABLES / f"{task}_shap_top.csv")
        top = imp.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        left = np.zeros(len(top))
        cls_cols = [c for c in imp.columns if c != "total"]
        cmap = colors_for(cls_cols) if len(cls_cols) > 1 else {cls_cols[0]: SERIES[0]}
        for c in cls_cols:
            ax.barh(top.index, top[c], left=left, color=cmap[c], label=c, height=0.65)
            left += top[c].to_numpy()
        ax.set_xlabel("mean |SHAP value| (contribution to predicted probability)")
        ax.set_title(f"Random forest — top-15 features by SHAP ({task})")
        if len(cls_cols) > 1:
            ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", visible=False)
        save(fig, f"eval_{task}_shap")

    if "logreg" in models:
        pipe = models["logreg"]
        names = list(pipe.named_steps["feat"].get_feature_names_out())
        coef = pipe.named_steps["model"].coef_
        if coef.shape[1] == len(names):
            rows = []
            cls = d.classes if coef.shape[0] == len(d.classes) else [d.classes[1]]
            for i, c in enumerate(cls):
                s = pd.Series(coef[i], index=names)
                for f in s.abs().sort_values(ascending=False).index[:12]:
                    rows.append({"class": c, "feature": f, "coef": s[f]})
            out = pd.DataFrame(rows)
            out["n_nonzero_total"] = int((np.abs(coef).max(0) > 1e-6).sum())
            out.round(4).to_csv(TABLES / f"{task}_logreg_top.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[TASK_A, TASK_C, TASK_B])
    a = ap.parse_args()
    for t in a.tasks:
        if (MODELS_DIR / t).exists():
            evaluate_task(t)


if __name__ == "__main__":
    main()
