"""Train, tune and cross-validate every model for one or more tasks.

Protocol (training split only — the 20 % test split is never touched here)
  1. feature ablation         which derived feature groups help (5-fold CV)
  2. default CV               every model with default hyperparameters (5-fold CV)
  3. tuning                   RandomizedSearchCV, 5-fold stratified CV, macro-F1
  4. ensembles                soft voting over the tuned [logreg, svm_rbf, random_forest]
  5. repeated CV              tuned models, 5-fold x 2 repeats -> per-fold scores for statistical comparison
  6. nested CV                outer 5-fold around the whole tuning procedure (unbiased estimate)
  7. final fit                every tuned model refit on the full training split -> models/<task>/*.joblib

Usage:  uv run glioma-train [--tasks subtype_3 grade_2] [--fast]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from glioma.build_dataset import TASK_A, TASK_B, TASK_C
from glioma.config import CV_FOLDS, ROOT, SEED, TABLES
from glioma.data import load_task
from glioma.features import GliomaFeatures, TabularFeatures
from glioma.models import (
    CACHE,
    CLASSICAL,
    ENSEMBLES,
    VOTING_MEMBERS,
    make_pipeline,
    soft_voting,
    specs,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*feature names.*")

MODELS_DIR = ROOT / "models"
PRIMARY = "f1_macro"


def scoring(n_classes: int) -> dict:
    s = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy",
         "neg_log_loss": "neg_log_loss"}
    s["roc_auc"] = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    return s


def cv5():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ 1. feature ablation
def feature_ablation(task, X, y) -> pd.DataFrame:
    if task == TASK_C:
        variants = {"raw (clinical + 20 mutation flags)": dict(derived=False),
                    "raw + derived combinations": dict(derived=True)}
        make_feat = lambda kw: TabularFeatures(**kw)  # noqa: E731
    else:
        variants = {
            "genes only (top-2000)": dict(use_arms=False, use_sets=False, use_clinical=False),
            "arm scores only": dict(use_genes=False, use_sets=False, use_clinical=False),
            "pathway scores only": dict(use_genes=False, use_arms=False, use_clinical=False),
            "derived only (arms+pathways+clinical)": dict(use_genes=False),
            "all (genes+arms+pathways+clinical)": dict(),
        }
        make_feat = lambda kw: GliomaFeatures(**kw)  # noqa: E731
    rows = []
    for name in ["logreg", "random_forest"]:
        for vname, kw in variants.items():
            pipe = make_pipeline(task, name)
            pipe.set_params(feat=make_feat(kw))
            r = cross_validate(pipe, X, y, cv=cv5(), scoring=scoring(y.nunique()), n_jobs=5)
            rows.append({"model": name, "features": vname,
                         "n_features": len(make_feat(kw).fit(X).feature_names_),
                         **{k[5:]: r[k].mean() for k in r if k.startswith("test_")},
                         "f1_macro_sd": r["test_f1_macro"].std()})
            log(f"  ablation {name:14s} {vname:40s} F1={rows[-1]['f1_macro']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"{task}_feature_ablation.csv", index=False)
    return df


# ------------------------------------------------------------------ 3. tuning
def tune_random(task, name, X, y, n_iter=None) -> tuple[dict, pd.DataFrame]:
    spec = specs(task)[name]
    search = RandomizedSearchCV(make_pipeline(task, name), spec.space, n_iter=n_iter or spec.n_iter,
                                scoring=PRIMARY, cv=cv5(), n_jobs=8, random_state=SEED,
                                return_train_score=True, error_score="raise")
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    return search.best_params_, res


# ------------------------------------------------------------------ main loop per task
def run_task(task: str, fast: bool = False) -> None:
    t0 = time.time()
    d = load_task(task)
    enc = {c: i for i, c in enumerate(d.classes)}
    X, y = d.X_train, d.y_train.map(enc).astype(int)
    out_dir = MODELS_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = scoring(len(d.classes))
    log(f"=== {task}: {d.description} | train {X.shape}, test {d.X_test.shape}")

    log("1. feature ablation")
    feature_ablation(task, X, y)

    log("2. default-hyperparameter CV")
    names = CLASSICAL + [n for n in ENSEMBLES if n != "soft_voting"]
    rows = []
    for name in names:
        r = cross_validate(make_pipeline(task, name), X, y, cv=cv5(), scoring=sc, n_jobs=5,
                           return_train_score=True)
        rows.append({"model": name, **{k: r[k].mean() for k in r if k.startswith(("test_", "train_"))}})
        log(f"  default {name:14s} F1={rows[-1]['test_f1_macro']:.3f} (train {rows[-1]['train_f1_macro']:.3f})")
    pd.DataFrame(rows).to_csv(TABLES / f"{task}_cv_default.csv", index=False)

    log("3. hyperparameter tuning")
    best_params, tuning_rows = {}, []
    for name in names:
        ts = time.time()
        params, res = tune_random(task, name, X, y, n_iter=4 if fast else None)
        keep = [c for c in res.columns if c.startswith(("param_", "mean_", "std_", "rank_"))]
        res[keep].to_csv(TABLES / f"{task}_search_{name}.csv", index=False)
        best_score = res["mean_test_score"].max()
        n_tried = len(res)
        best_params[name] = params
        tuning_rows.append({"model": name, "method": "random search",
                            "candidates": n_tried, "best_cv_f1_macro": best_score,
                            "seconds": round(time.time() - ts), "best_params": json.dumps(params, default=str)})
        log(f"  tuned  {name:14s} best F1={best_score:.3f} ({n_tried} candidates, {time.time() - ts:.0f}s)")
    pd.DataFrame(tuning_rows).to_csv(TABLES / f"{task}_tuning_best.csv", index=False)
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=1, default=str))

    tuned = {n: make_pipeline(task, n).set_params(**p) for n, p in best_params.items()}
    members = {n: clone(tuned[n]) for n in VOTING_MEMBERS}
    tuned["soft_voting"] = soft_voting(members)

    log("4-5. ensembles + repeated CV of tuned models (5-fold x 2)")
    rcv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=1 if fast else 2, random_state=SEED)
    fold_rows = []
    for name, pipe in tuned.items():
        r = cross_validate(pipe, X, y, cv=rcv, scoring=sc, n_jobs=5, return_train_score=True)
        for f in range(len(r["test_f1_macro"])):
            scores = {k: r[k][f] for k in r if k.startswith(("test_", "train_"))}
            fold_rows.append({"model": name, "fold": f, **scores})
        log(f"  tuned  {name:14s} CV F1={r['test_f1_macro'].mean():.3f} ± {r['test_f1_macro'].std():.3f}")
    pd.DataFrame(fold_rows).to_csv(TABLES / f"{task}_cv_tuned_folds.csv", index=False)

    if task in (TASK_A, TASK_C) and not fast:
        log("6. nested CV (outer 5-fold around random search)")
        nested_cv(task, X, y)

    log("7. refit tuned models on full training split")
    for name, pipe in tuned.items():
        pipe.fit(X, y)
        joblib.dump(pipe, out_dir / f"{name}.joblib")
    (out_dir / "classes.json").write_text(json.dumps(d.classes))
    CACHE.clear(warn=False)  # the transformer cache only helps within a run
    log(f"=== {task} done in {(time.time() - t0) / 60:.1f} min")


def nested_cv(task, X, y) -> None:
    outer = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + 1)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows = []
    for name in ["logreg", "svm_rbf", "random_forest"]:
        spec = specs(task)[name]
        for f, (tr, te) in enumerate(outer.split(X, y)):
            search = RandomizedSearchCV(make_pipeline(task, name), spec.space, n_iter=10, scoring=PRIMARY,
                                        cv=inner, n_jobs=8, random_state=SEED)
            search.fit(X.iloc[tr], y.iloc[tr])
            pred = search.predict(X.iloc[te])
            rows.append({"model": name, "outer_fold": f, "inner_best_f1": search.best_score_,
                         "outer_f1_macro": f1_score(y.iloc[te], pred, average="macro"),
                         "outer_balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
                         "chosen_params": json.dumps(search.best_params_, default=str)})
        sub = pd.DataFrame(rows).query("model == @name")
        log(f"  nested {name:14s} outer F1={sub.outer_f1_macro.mean():.3f} ± {sub.outer_f1_macro.std():.3f} "
            f"(inner optimistic {sub.inner_best_f1.mean():.3f})")
    pd.DataFrame(rows).to_csv(TABLES / f"{task}_nested_cv.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[TASK_A, TASK_C, TASK_B])
    ap.add_argument("--fast", action="store_true", help="tiny search budgets (smoke test)")
    a = ap.parse_args()
    for t in a.tasks:
        run_task(t, fast=a.fast)


if __name__ == "__main__":
    main()
