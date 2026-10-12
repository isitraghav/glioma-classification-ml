"""External validation of the Task-A subtype model on independent NCBI GEO cohorts.

The TCGA models were trained on RNA-seq; both GEO cohorts are microarrays from other hospitals.
To make the expression scales comparable we
  1. restrict every cohort to the genes it shares with TCGA,
  2. replace each sample's values by its within-sample percentile rank (SampleRankNormalizer) —
     a stateless transform, so nothing is fitted on the external data,
and retrain the tuned Task-A pipelines on the rank-normalised TCGA training split.

* GSE107850 (EORTC 22033-26033 trial, 195 grade-II gliomas, IDH status known):
  IDH-mutant vs IDH-wildtype accuracy, where P(IDH-mutant) = P(non-codel) + P(codel).
* GSE16011 (Gravendeel 2009, 276 gliomas, histology only):
  concordance between predicted molecular class and histology (GBM -> IDHwt, oligodendroglioma -> codel).
"""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from glioma.build_dataset import TASK_A
from glioma.config import PROCESSED, TABLES
from glioma.data import load_task
from glioma.features import SampleRankNormalizer
from glioma.models import make_pipeline
from glioma.plots import SERIES, save
from glioma.train import MODELS_DIR

warnings.filterwarnings("ignore")
MODELS = ["logreg", "svm_rbf", "random_forest"]


def _prep(expr: pd.DataFrame, meta: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    X = expr.reindex(columns=genes)
    X = X.fillna(X.median())
    X["age"] = pd.to_numeric(meta["age"], errors="coerce").reindex(X.index)
    sex = meta.get("gender", pd.Series(index=meta.index, dtype=object)).reindex(X.index)
    X["male"] = (sex.str.lower() == "male").astype("float32")
    return X


def rank_pipeline(name: str, params: dict) -> Pipeline:
    pipe = make_pipeline(TASK_A, name).set_params(**params)
    pipe.memory = None
    return Pipeline([("rank", SampleRankNormalizer())] + pipe.steps)


def main() -> None:
    d = load_task(TASK_A)
    best = json.loads((MODELS_DIR / TASK_A / "best_params.json").read_text())
    enc = {c: i for i, c in enumerate(d.classes)}
    cohorts = {g: (pd.read_parquet(PROCESSED / f"ext_{g}_expr.parquet"),
                   pd.read_parquet(PROCESSED / f"ext_{g}_meta.parquet")) for g in ["gse107850", "gse16011"]}
    rows, conc_rows = [], []
    for g, (e, m) in cohorts.items():
        genes = sorted(set(e.columns) & set(d.X_train.columns) - {"age", "male"})
        Xtr = d.X_train[genes + ["age", "male"]]
        Xte_int = d.X_test[genes + ["age", "male"]]
        Xext = _prep(e, m, genes)
        for name in MODELS:
            pipe = rank_pipeline(name, best[name]).fit(Xtr, d.y_train.map(enc))
            # sanity: rank-normalised model still works on TCGA test
            p_int = pipe.predict_proba(Xte_int)
            internal_f1 = f1_score(d.y_test.map(enc), p_int.argmax(1), average="macro")
            P = pipe.predict_proba(Xext)
            pred = pd.Series([d.classes[i] for i in P.argmax(1)], index=Xext.index)
            row = {"cohort": g, "model": name, "shared_genes": len(genes), "n": len(Xext),
                   "tcga_test_f1_rank_normalised": internal_f1}
            if g == "gse107850":
                known = m["idh_status"].notna()
                ytrue = (m.loc[known, "idh_status"] == "Mutant").astype(int)
                p_mut = pd.Series(P[:, enc["IDHmut-non-codel"]] + P[:, enc["IDHmut-codel"]], index=Xext.index)[known]
                yhat = (p_mut > 0.5).astype(int)
                row.update({"n_idh_known": int(known.sum()), "n_idh_wt": int((ytrue == 0).sum()),
                            "idh_auc": roc_auc_score(ytrue, p_mut), "idh_accuracy": float((yhat == ytrue).mean()),
                            "idh_balanced_accuracy": balanced_accuracy_score(ytrue, yhat),
                            "idh_sensitivity_mut": float(yhat[ytrue == 1].mean()),
                            "idh_specificity_wt": float(1 - yhat[ytrue == 0].mean())})
            ct = pd.crosstab(m["histology"].reindex(pred.index), pred, normalize="index").reindex(columns=d.classes,
                                                                                                fill_value=0)
            for h, r in ct.iterrows():
                conc_rows.append({"cohort": g, "model": name, "histology": h,
                                  "n": int((m["histology"] == h).sum()), **r.round(3).to_dict()})
            if g == "gse16011":
                gbm = m["histology"] == "glioblastoma"
                od = m["histology"] == "oligodendroglioma"
                row.update({"gbm_pred_idhwt": float((pred[gbm] == "IDHwt").mean()),
                            "oligo_pred_codel": float((pred[od] == "IDHmut-codel").mean())})
            rows.append(row)
            print(row, flush=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(TABLES / "external_validation.csv", index=False)
    conc = pd.DataFrame(conc_rows)
    conc.to_csv(TABLES / "external_histology_concordance.csv", index=False)
    plot(conc)


def plot(conc: pd.DataFrame) -> None:
    sub = conc[conc["model"] == "logreg"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 5]})
    classes = ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"]
    for ax, g, title in zip(axes, ["gse107850", "gse16011"],
                            ["GSE107850 (EORTC trial, grade II)", "GSE16011 (Gravendeel, all grades)"]):
        t = sub[sub["cohort"] == g].set_index("histology")
        t = t[t["n"] >= 5]
        left = np.zeros(len(t))
        for c, colr in zip(classes, SERIES):
            ax.barh([f"{h} (n={n})" for h, n in zip(t.index, t["n"])], t[c], left=left, color=colr, label=c,
                    height=0.62, edgecolor="#fcfcfb", lw=1)
            left += t[c].to_numpy()
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of patients")
        ax.set_title(f"{title}: predicted class by histology")
        ax.grid(axis="y", visible=False)
    axes[1].legend(fontsize=7.5, loc="lower right")
    save(fig, "external_histology_concordance")


if __name__ == "__main__":
    main()
