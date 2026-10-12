"""Predict glioma type for new tumour samples with a trained pipeline.

Input: a CSV/TSV with one row per sample and one column per gene (official NCBI symbols), holding
TCGA-style RNA-seq expression. Values may be RSEM/TPM-like (raw) or already log2(x+1); raw input is
detected and log-transformed. Optional columns `age` (years) and `sex` (male/female) are used if present.

Output: one row per sample with the predicted type, its plain-language name and every class probability
(plus the top 3 features that pushed each prediction: SHAP for tree models, coefficient × value for
logistic regression).

Usage
  uv run glioma-predict --make-example example.tsv          # write 5 held-out TCGA patients as a demo input
  uv run glioma-predict --input example.tsv --out preds.csv  # predict with the model selected by CV
  uv run glioma-predict --input x.tsv --model logreg         # or any other saved model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from glioma.build_dataset import TASK_A
from glioma.config import TABLES
from glioma.train import MODELS_DIR

PLAIN_NAMES = {
    "IDHwt": "Glioblastoma (IDH-wildtype)",
    "IDHmut-non-codel": "Astrocytoma (IDH-mutant, 1p/19q intact)",
    "IDHmut-codel": "Oligodendroglioma (IDH-mutant, 1p/19q-codeleted)",
    "LGG": "Lower-grade glioma", "GBM": "Glioblastoma",
}
MAX_MISSING_GENES = 0.20


def selected_model(task: str) -> str:
    f = TABLES / f"{task}_best_model.json"
    if not f.exists():
        raise SystemExit(f"no evaluated model for {task}: run `glioma-evaluate` first")
    return json.loads(f.read_text())["best_by_cv"]


def read_input(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df.columns = df.columns.astype(str)
    return df


def prepare(df: pd.DataFrame, pipe) -> tuple[pd.DataFrame, dict]:
    """Align columns to what the pipeline was trained on; fill missing genes with the training mean."""
    feat = pipe.named_steps["feat"]
    genes = list(feat.genes_)
    sex = df.pop("sex") if "sex" in df else None
    age = pd.to_numeric(df.pop("age"), errors="coerce") if "age" in df else pd.Series(np.nan, index=df.index)
    expr = df.apply(pd.to_numeric, errors="coerce")
    info = {"log2_applied": False}
    if np.nanmax(expr.to_numpy()) > 50:  # raw RSEM/TPM-like values -> same scale as training
        expr = np.log2(expr.clip(lower=0) + 1)
        info["log2_applied"] = True
    missing = [g for g in genes if g not in expr.columns]
    info["genes_expected"], info["genes_missing"] = len(genes), len(missing)
    if len(missing) / len(genes) > MAX_MISSING_GENES:
        raise SystemExit(f"{len(missing)}/{len(genes)} model genes are missing from the input "
                         f"(>{MAX_MISSING_GENES:.0%}); check that columns are official gene symbols")
    fill = pd.DataFrame(np.tile(feat.mu_[[genes.index(g) for g in missing]], (len(expr), 1)),
                        index=expr.index, columns=missing)
    X = pd.concat([expr.reindex(columns=[g for g in genes if g in expr.columns]), fill], axis=1)[genes]
    X["age"] = age.to_numpy()
    X["male"] = (sex.astype(str).str.lower().str.startswith("m")).astype("float32").to_numpy() \
        if sex is not None else np.nan
    return X, info


def drivers(pipe, X: pd.DataFrame, pred_idx: np.ndarray, top: int = 3) -> list[str] | None:
    """Top features pushing each sample towards its predicted class (None if the model has no explainer)."""
    model = pipe.named_steps["model"]
    Xt = np.asarray(pipe[:-1].transform(X))
    names = list(pipe.named_steps["feat"].get_feature_names_out())
    if Xt.shape[1] != len(names):  # a PCA step makes features uninterpretable
        return None
    if hasattr(model, "estimators_") and hasattr(model, "feature_importances_"):  # tree ensemble / tree
        import shap
        sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
        if sv.ndim == 3 and sv.shape[0] != Xt.shape[0]:  # (classes, samples, features) -> (samples, features, classes)
            sv = np.transpose(sv, (1, 2, 0))
        contrib = lambda i, k: sv[i, :, k] if sv.ndim == 3 else sv[i]  # noqa: E731
    elif hasattr(model, "coef_") and model.coef_.shape[1] == Xt.shape[1]:  # linear: coefficient x value
        coef = model.coef_ if model.coef_.shape[0] > 1 else np.vstack([-model.coef_, model.coef_])
        contrib = lambda i, k: coef[k] * Xt[i]  # noqa: E731
    else:
        return None
    out = []
    for i, k in enumerate(pred_idx):
        c = contrib(i, k)
        order = np.argsort(c)[::-1][:top]
        out.append("; ".join(f"{names[j]} (+{c[j]:.2f})" for j in order))
    return out


def predict(df: pd.DataFrame, task: str = TASK_A, model: str | None = None) -> tuple[pd.DataFrame, dict]:
    name = model or selected_model(task)
    pipe = joblib.load(MODELS_DIR / task / f"{name}.joblib")
    classes = json.loads((MODELS_DIR / task / "classes.json").read_text())
    X, info = prepare(df, pipe)
    proba = pipe.predict_proba(X)
    idx = proba.argmax(1)
    out = pd.DataFrame({"predicted_label": [classes[i] for i in idx],
                        "predicted_type": [PLAIN_NAMES.get(classes[i], classes[i]) for i in idx],
                        "confidence": proba.max(1).round(4)}, index=X.index)
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i].round(4)
    top = drivers(pipe, X, idx)
    if top is not None:
        out["top_drivers"] = top
    info.update({"task": task, "model": name, "n_samples": len(out)})
    return out, info


def make_example(path: Path, n: int = 5) -> None:
    """Write n held-out TCGA test patients (never seen in training) as a demo input file."""
    from glioma.data import load_task, meta
    d = load_task(TASK_A)
    ids = d.y_test.groupby(d.y_test).head(2).index[:n]
    X = d.X_test.loc[ids].drop(columns=["age", "male"])
    X.insert(0, "sex", meta().loc[ids, "gender"])
    X.insert(0, "age", meta().loc[ids, "age"])
    X.to_csv(path, sep="\t")
    truth = d.y_test.loc[ids].map(PLAIN_NAMES)
    print(f"wrote {path} ({len(ids)} samples x {X.shape[1] - 2} genes + age + sex). True types:")
    print(truth.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict glioma type from tumour gene expression.")
    ap.add_argument("--input", type=Path, help="CSV/TSV: rows = samples, columns = gene symbols [+ age, sex]")
    ap.add_argument("--out", type=Path, help="write predictions to this CSV (default: print)")
    ap.add_argument("--task", default=TASK_A)
    ap.add_argument("--model", help="saved model name (default: the model selected by CV)")
    ap.add_argument("--make-example", type=Path, metavar="PATH", help="write a demo input file and exit")
    a = ap.parse_args()
    if a.make_example:
        make_example(a.make_example)
        return
    if not a.input:
        ap.error("--input is required (or use --make-example)")
    out, info = predict(read_input(a.input), a.task, a.model)
    print(f"model: {info['model']} | samples: {info['n_samples']} | genes missing: "
          f"{info['genes_missing']}/{info['genes_expected']} | log2 applied: {info['log2_applied']}",
          file=sys.stderr)
    if a.out:
        out.to_csv(a.out)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(out.drop(columns=[c for c in out if c.startswith("p_")]).to_string())


if __name__ == "__main__":
    main()
