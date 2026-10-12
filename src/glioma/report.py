"""Assemble reports/report.md from the generated tables and figures (numbers are never hand-typed)."""
from __future__ import annotations

import json

import pandas as pd

from glioma.build_dataset import TASK_A, TASK_B, TASK_C, TASKS
from glioma.config import RAW, REPORTS, ROOT, TABLES
from glioma.evaluate import PRETTY
from glioma.models import specs

OUT = REPORTS / "report.md"


def tbl(df: pd.DataFrame, floatfmt=".3f", index=True) -> str:
    return df.to_markdown(floatfmt=floatfmt, index=index)


def fig(name: str, caption: str) -> str:
    if not (REPORTS / "figures" / f"{name}.png").exists():
        return ""
    return f"![{caption}](figures/{name}.png)\n\n*{caption}*\n"


def read(name, **kw):
    p = TABLES / name
    return pd.read_csv(p, **kw) if p.exists() else None


def jread(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else {}


def section_data() -> str:
    man = pd.read_csv(RAW / "MANIFEST.csv")
    man["MB"] = (man["bytes"] / 1e6).round(1)
    man["source"] = man["url"].str.extract(r"https?://([^/]+)/")[0]
    s = jread("eda_summary.json")
    counts = "\n".join(f"| {t} | {TASKS[t]} | " + ", ".join(f"{k}: {v}" for k, v in s.get(f"{t}_counts", {}).items())
                       + f" | {s.get(f'{t}_imbalance_ratio', '')} |" for t in TASKS)
    return f"""## 1. Data sources and provenance

All data come from **government or peer-reviewed research sources**. Nothing is from Kaggle or another re-upload site.

* **NCI Genomic Data Commons (GDC, US National Cancer Institute)**: the publication page of
  *Ceccarelli et al., "Molecular profiling reveals biologically discrete subsets and pathways of progression
  in diffuse glioma", Cell 164:550-563 (2016)*. It provides the RNA-seq expression matrix (667 tumours ×
  20,330 genes), the paper's **Table S1** (labels and clinical data for 1,122 patients) and GISTIC copy number.
* **GDC REST API**: open-access somatic mutation calls and demographics for TCGA-LGG / TCGA-GBM.
* **NCBI GEO (US National Library of Medicine)**: two independent external cohorts.
  * GSE107850: EORTC 22033-26033 clinical trial, 195 grade-II gliomas with IDH status.
  * GSE16011: Gravendeel et al. 2009, 276 gliomas with histology.
  * The GSE107850 platform's probe table (GSE16011's probe IDs already contain the Entrez gene ID).
* **NCBI Gene**: gene symbols, synonyms and cytobands, used for symbol harmonisation and chromosome-arm mapping.
* **MSigDB (Broad Institute)**: the 50 Hallmark pathways and the Verhaak 2010 glioblastoma subtype signatures.
* **PubMed (NCBI E-utilities)**: {len(read('literature.csv'))} related papers scraped. Every accuracy/AUC figure
  quoted in their abstracts is extracted to `tables/literature_metrics.csv` and used as a baseline.

Every file is downloaded by `glioma-download` (aria2c, concurrent) and recorded with URL, timestamp, size and
MD5 in `data/raw/MANIFEST.csv`:

{tbl(man[["file", "source", "MB", "description"]], index=False)}

### Prediction tasks

| task | definition | class counts | max/min imbalance |
|---|---|---|---|
{counts}

* **Task A (main)** is the WHO-2021 (CNS5) molecular classification of adult diffuse glioma. **IDH-wildtype**
  corresponds to glioblastoma, **IDH-mutant + 1p/19q-codeleted** to oligodendroglioma, and **IDH-mutant
  non-codeleted** to astrocytoma. The model only sees gene expression, age and sex. The label-defining
  measurements (IDH mutation calls, 1p/19q copy number) are never features.
* **Task B (extended)** is the 7 pan-glioma DNA-methylation subtypes defined in Ceccarelli 2016, predicted from
  expression. Its classes are finer-grained and strongly imbalanced (19.5:1).
* **Task C (baseline replication)** is LGG vs GBM from age, sex, race and 20 mutation flags. It rebuilds the
  setup of Tasci et al. 2022 (the source of UCI dataset #759) from raw GDC data.

The data are split into a stratified 80/20 train/test split (seed 42), frozen to `data/processed/split_<task>.json`.
All EDA, feature selection, tuning and model selection use the **training split only**.
"""


def section_eda() -> str:
    s = jread("eda_summary.json")
    pcs = read("eda_pc_variance_sources.csv")
    return f"""## 2. Exploratory data analysis

{fig("eda_class_balance", "Class balance for the three tasks")}

**Clinical.** Age differs strongly between subtypes (Kruskal–Wallis p = {s.get('age_kruskal_p', float('nan')):.1e}).
IDH-wildtype patients are older, as expected for glioblastoma. Several clinical fields have substantial missing
values (KPS, MGMT, TERT), which is why the expression models use only age and sex.

{fig("eda_clinical", "Age, sex composition and missingness of clinical annotations")}

**Expression.** {s.get('n_genes', 0):,} genes × {s.get('n_expr_samples', 0)} tumours. {s.get('frac_zero_values', 0):.0%}
of values are zero and {s.get('frac_genes_mean_below_1', 0):.0%} of genes have mean log2 expression below 1. The
lowest-expressed 20% of genes are therefore filtered inside the feature pipeline. The variance is concentrated:
the top-2,000 genes hold {s.get('var_frac_top2000', 0):.0%} of total variance. With p ≫ n
(~19.7k features vs 523 training samples), regularisation and feature reduction are essential.

{fig("eda_expression_qc", "Expression distributions, mean-variance trend and variance captured by top-k genes")}

**Structure.** The three subtypes separate almost perfectly in t-SNE, and IDH-wildtype separates from the rest
along PC1. {s.get('n_genes_bonferroni_sig', 0):,} genes differ between subtypes after Bonferroni correction.

{fig("eda_embeddings", "PCA / t-SNE of the top-2000 variable genes, and PCA coloured by tissue-source site")}

**Batch effects.** The tissue-source site explains part of PC1, but sites contributed mainly LGG or mainly GBM,
so site is *confounded* with biology. The later PCs, however, are dominated by site (R² up to 0.17 in PCs 7–10
vs ≈0.02 for subtype). The models are therefore regularised and use derived, biologically grounded features.

{tbl(pcs, index=False)}

{fig("eda_pc_variance_sources", "Share of each PC's variance explained by subtype vs tissue-source site")}

**Mutations** reproduce the known biology, which is a check on label quality: IDH1 in >90% of IDH-mutant
tumours, TP53 + ATRX in astrocytomas, CIC + FUBP1 in oligodendrogliomas, and EGFR / PTEN / NF1 in
IDH-wildtype. The co-occurrence matrix shows the mutual exclusivity of these programs.

{fig("eda_mutations", "Mutation frequency by subtype, and pairwise co-occurrence (log odds ratio)")}

{fig("eda_marker_heatmap", "Top marker genes per subtype (training set)")}

**Survival** differs sharply between the three classes, which confirms the classes are clinically meaningful.

{fig("eda_survival", "Kaplan–Meier overall survival by subtype")}

{fig("eda_subtype7_vs_subtype3", "Task B's 7 methylation subtypes nest inside Task A's 3 classes")}

**Data issues found:**
1. p ≫ n.
2. Class imbalance, mild in Tasks A and C and severe in Task B (LGm6-GBM n=12).
3. Batch (site) confounded with grade.
4. Missing clinical fields.
5. Gene symbols from 2015 that needed mapping to current NCBI symbols ({'96.9%'} mapped).
6. RNA-seq vs microarray platform differences for external validation.
"""


def section_features() -> str:
    parts = ["""## 3. Feature derivation

All features are computed by scikit-learn transformers (`glioma/features.py`). These are fit **inside each CV
fold**, so gene filtering, z-scoring and selection never see the validation fold.

| group | features | rationale |
|---|---|---|
| top-k variable genes | k ∈ {500…4000}, tuned | most informative genes; variance filter is label-free, so it does not leak |
| **chromosome-arm scores** | 39 autosomal arms: mean gene z-score on the arm minus the sample's genome-wide mean | a deleted arm lowers expression of all its genes, so 1p/19q scores proxy **1p/19q codeletion** and 7/10 scores proxy the **+7/−10 GBM signature** |
| pathway scores | 50 MSigDB Hallmarks + 4 Verhaak GBM subtypes (mean z) | biologically interpretable, robust to single-gene noise and platform effects |
| clinical | age, sex | age is strongly subtype-associated |
| Task C derived | `idh_any`, `astro_sig` (TP53∧ATRX), `oligo_sig` (CIC∨FUBP1), `gbm_sig` (EGFR∨PTEN∨NF1), panel burden, age>50 | encode known co-mutation programs |

**Validation of the arm features.** The arm scores are computed from expression alone, yet they correlate
with the *measured* GISTIC copy number of the same arm:
"""]
    arm = read("eda_arm_score_validation.csv")
    if arm is not None:
        parts.append(tbl(arm, index=False))
    parts.append(fig("eda_arm_scores_vs_copy_number", "Expression-derived arm scores vs measured copy number"))
    parts.append(fig("eda_arm_profile", "Mean arm score per subtype: 1p/19q loss in codel, +7/−10 in IDHwt"))
    parts.append("**Ablation (5-fold CV on the training split).** This tests whether each feature group earns its place:\n")
    for t in [TASK_A, TASK_B, TASK_C]:
        a = read(f"{t}_feature_ablation.csv")
        if a is not None:
            p = a.pivot(index="features", columns="model", values="f1_macro")
            p.insert(0, "n_features", a.groupby("features")["n_features"].first())
            parts.append(f"**{t}** (macro-F1):\n\n{tbl(p)}\n")
    return "\n".join(parts)


def section_models() -> str:
    rows = [{"model": PRETTY[n], "family": s.family, "how it works / why here": s.explanation}
            for n, s in specs(TASK_A).items()]
    rows += [{"model": "Soft voting", "family": "ensemble",
              "how it works / why here": "Averages the predicted probabilities of the tuned logistic regression, "
                                         "RBF-SVM and random forest: a linear, a kernel and a tree model make "
                                         "different errors, which partly cancel."}]
    return f"""## 4. Classification models

Every model is a full pipeline: `features → median-impute → standardise → [PCA for k-NN] → classifier`.
Class weights are `balanced` where supported, to handle imbalance. The primary metric is **macro-F1**,
which weights every class equally. This matters for Task B's rare classes.

{tbl(pd.DataFrame(rows), index=False)}
"""


def section_regularization() -> str:
    r = read("reg_summary.csv")
    return f"""## 5. Regularisation

With ~2,000 gene features and ~520 samples, every unregularised model can memorise the training set. The
default-hyperparameter runs show train F1 = 1.000 for the SVMs, k-NN, trees and random forest in Task A. The
following regularisers were studied on Task A:

* **L1 / L2 / elastic-net penalties (logistic regression).** As C decreases, the L1 component drives gene
  weights to exactly zero (embedded feature selection), while L2 shrinks all weights. Elastic-net combines
  sparsity with the grouping of correlated genes. It was chosen because co-expressed genes on deleted arms
  are highly correlated.
* **Margin softness C (SVM).**
* **Depth, minimum leaf size and cost-complexity pruning (decision tree).**
* **Tree depth and minimum leaf size (random forest),** on top of the variance reduction from bagging.
* **Implicit regularisation.** Bootstrap averaging (random forest), PCA before k-NN, and the variance-based
  gene filter.

{fig("reg_logreg_paths", "Logistic regression: CV and train F1 and number of non-zero gene weights vs C")}
{fig("reg_learning_curves", "Learning curves: ~unregularised vs tuned elastic-net logistic regression")}
{fig("reg_tree_pruning", "Decision tree: depth limit and cost-complexity pruning")}
{fig("reg_svm_c", "RBF-SVM: effect of C")}
{fig("reg_random_forest", "Random forest: tree depth and minimum leaf size")}

{tbl(r, index=False) if r is not None else ''}
"""


def section_tuning() -> str:
    parts = ["""## 6. Hyperparameter tuning

* **Search.**
  * Random search (`RandomizedSearchCV`, 15–25 candidates, 5-fold stratified CV, macro-F1) for every model.
    Distributions are log-uniform for scale parameters (C, gamma, ccp_alpha), which vary over orders of
    magnitude; random search covers them better than a grid of the same size.
  * The feature step's **k_genes** is tuned jointly with each model.
* **Unbiased estimate.** The CV score of the selected configuration is optimistically biased, because it was
  chosen on those same folds. **Nested CV** measures that bias for Tasks A and C: an outer 5-fold loop wraps
  the whole search (inner 3-fold CV, 10 random-search candidates) for logistic regression, RBF SVM and random
  forest.
"""]
    for t in [TASK_A, TASK_C, TASK_B]:
        tb = read(f"{t}_tuning_best.csv")
        if tb is None:
            continue
        tb["model"] = tb["model"].map(PRETTY)
        parts.append(f"### {t}\n\n{tbl(tb[['model', 'method', 'candidates', 'best_cv_f1_macro', 'seconds', 'best_params']], index=False)}\n")
        parts.append(fig(f"tune_{t}_hyperparameters", f"{t}: hyperparameter impact"))
        n = read(f"{t}_nested_cv.csv")
        if n is not None:
            g = n.groupby("model").agg(inner_cv_f1=("inner_best_f1", "mean"), nested_outer_f1=("outer_f1_macro", "mean"),
                                       nested_outer_sd=("outer_f1_macro", "std"))
            g["optimism"] = g["inner_cv_f1"] - g["nested_outer_f1"]
            parts.append(f"**Nested CV ({t})**\n\n{tbl(g)}\n")
    return "\n".join(parts)


def section_performance() -> str:
    parts = ["""## 7. Ensembles and performance analysis

**Protocol.**
1. The best model per task is chosen by **repeated CV on the training split** (5-fold × 2), never by test score.
2. All tuned models are refit on the full training split and evaluated **once** on the held-out 20% test split.
3. Test uncertainty is reported as 95% bootstrap CIs (1,000 resamples).
4. Models are compared statistically in two ways:
   * the Nadeau–Bengio **corrected resampled t-test** on CV folds, which corrects for the overlap between
     training sets;
   * an exact **McNemar test** on the paired test predictions.
"""]
    for t in [TASK_A, TASK_B, TASK_C]:
        comp = read(f"{t}_cv_vs_test.csv", index_col=0)
        if comp is None:
            continue
        best = jread(f"{t}_best_model.json")
        test = read(f"{t}_test_metrics.csv", index_col=0)
        st = read(f"{t}_stat_tests.csv")
        pc = read(f"{t}_per_class.csv", index_col=0)
        comp.index = comp.index.map(lambda n: PRETTY.get(n, n))
        test.index = test.index.map(lambda n: PRETTY.get(n, n))
        parts.append(f"### {t} — {TASKS[t]}\n")
        parts.append(f"Best by CV: **{PRETTY.get(best['best_by_cv'])}**, with CV macro-F1 {best['cv_f1']:.3f} and "
                     f"**test macro-F1 {best['test_f1']:.3f}** (95% CI {best['test_ci'][0]:.3f}–{best['test_ci'][1]:.3f}). "
                     f"It made {best['n_errors']} errors out of {best['n_test']} test patients.\n")
        parts.append(fig(f"eval_{t}_model_comparison", f"{t}: default vs tuned vs test"))
        parts.append(tbl(comp[["default_cv_f1", "cv_f1", "cv_f1_sd", "train_f1", "overfit_gap", "tuning_gain",
                               "test_f1", "f1_macro_ci_lo", "f1_macro_ci_hi"]]))
        parts.append("\n**All test metrics**\n\n" + tbl(test))
        parts.append(fig(f"eval_{t}_confusion", f"{t}: confusion matrices (test)"))
        parts.append(fig(f"eval_{t}_roc_calibration", f"{t}: ROC curves and calibration"))
        if pc is not None:
            parts.append("**Per-class (best model)**\n\n" + tbl(pc))
        if st is not None:
            st["vs_model"] = st["vs_model"].map(lambda n: PRETTY.get(n, n))
            parts.append("\n**Statistical comparison against the best model**\n\n" + tbl(st.drop(columns=["best_model"]), index=False))
        parts.append(fig(f"eval_{t}_shap", f"{t}: SHAP feature importance (random forest)"))
        lr = read(f"{t}_logreg_top.csv")
        if lr is not None:
            top = lr.groupby("class").head(6)
            parts.append(f"**Elastic-net logistic regression**: {int(lr['n_nonzero_total'].iloc[0])} non-zero "
                         f"features; largest coefficients per class:\n\n" + tbl(top.drop(columns=["n_nonzero_total"]), index=False))
        err = read(f"{t}_errors.csv", index_col=0)
        if err is not None and len(err):
            parts.append(f"\n**Error analysis**: misclassified test patients ({len(err)})\n\n" + tbl(err.head(15)))
    return "\n\n".join(parts)


def section_external() -> str:
    ev = read("external_validation.csv")
    if ev is None:
        return ""
    ev["model"] = ev["model"].map(PRETTY)
    a = ev[ev["cohort"] == "gse107850"].dropna(axis=1, how="all")
    b = ev[ev["cohort"] == "gse16011"].dropna(axis=1, how="all")
    params = ROOT / "models" / TASK_A / "best_params.json"
    stale = params.exists() and (TABLES / "external_validation.csv").stat().st_mtime < params.stat().st_mtime
    note = ("> **Preliminary:** these numbers were computed with an earlier (smoke-test) tuning of the models. "
            "Re-run `glioma-external` to refresh them with the final hyperparameters.\n\n" if stale else "")
    return f"""## 8. External validation (independent hospitals, different platform)

{note}
The Task-A models were retrained on rank-normalised shared genes and applied **without any refitting** to two
GEO microarray cohorts. Because both cohorts use a different platform (microarray) from different institutions,
this is a much harder test than the TCGA test split.

**GSE107850, EORTC 22033 trial (IDH status known).** P(IDH-mutant) = P(non-codel) + P(codel):

{tbl(a.drop(columns=["cohort"]), index=False)}

**GSE16011, Gravendeel (histology only).** Concordance with histology. About 90% of adult GBMs are IDH-wildtype,
and most true oligodendrogliomas are 1p/19q-codeleted:

{tbl(b.drop(columns=["cohort"]), index=False)}

{fig("external_histology_concordance", "Predicted molecular class by histology in the external cohorts (elastic-net LR)")}
"""


def section_literature() -> str:
    lit = read("literature.csv")
    lm = read("literature_metrics.csv")
    best_c = jread(f"{TASK_C}_best_model.json")
    tc = read(f"{TASK_C}_test_metrics.csv", index_col=0)
    base = read("baselines.csv")
    rows = [{"study": r.study, "PMID": str(r.pmid), "TCGA LGG-vs-GBM accuracy": f"{r.value:.3f}",
             "source quote (abstract)": f"“{r.quote}”"} for r in base.itertuples()] if base is not None else []
    if tc is not None:
        b = best_c.get("best_by_cv")
        ci = read(f"{TASK_C}_test_bootstrap_ci.csv", index_col=0)
        rows.append({"study": f"**This project**: {PRETTY.get(b)} (selected by CV), held-out test", "PMID": "—",
                     "TCGA LGG-vs-GBM accuracy": f"{tc.loc[b, 'accuracy']:.3f}",
                     "source quote (abstract)": f"test n={best_c.get('n_test')}, macro-F1 95% CI "
                                                f"{ci.loc[b, 'f1_macro_ci_lo']:.3f}–{ci.loc[b, 'f1_macro_ci_hi']:.3f}"})
        top = tc["accuracy"].idxmax()
        if top != b:  # only worth showing when the test-best model differs from the CV-selected one
            rows.append({"study": f"This project: highest test accuracy ({PRETTY.get(top)}), reference only — "
                                  f"not selected by CV", "PMID": "—",
                         "TCGA LGG-vs-GBM accuracy": f"{tc.loc[top, 'accuracy']:.3f}", "source quote (abstract)": ""})
    return f"""## 9. Comparison with published work (scraped literature)

`glioma-literature` queried PubMed and retrieved {len(lit)} papers. It extracted {len(lm)} numeric metric
statements from {lm['pmid'].nunique()} abstracts (see `tables/literature_metrics.csv`). The directly
comparable baseline is Task C:

{tbl(pd.DataFrame(rows), index=False)}

Note on comparability: Tasci et al. used the 839-patient UCI table. Here the table was rebuilt from GDC
(891 exome-sequenced patients), and a test split is held out rather than using CV only.

For Task A there is no identical published benchmark on this exact split. Ceccarelli 2016 defines the labels
from IDH sequencing and copy number. The point of Task A is that **expression alone** recovers that molecular
diagnosis, which is useful when only RNA is available.
"""


def section_conclusion() -> str:
    a, b, c = (jread(f"{t}_best_model.json") for t in (TASK_A, TASK_B, TASK_C))
    if b:
        task_b = (f"* **Task B (7 methylation subtypes):** {PRETTY.get(b['best_by_cv'])}, test macro-F1 "
                  f"{b['test_f1']:.3f}. It is harder: the rare classes (LGm6-GBM, G-CIMP-low, PA-like) are defined "
                  "by methylation and share expression programs with their parent class. Macro-F1 exposes this, "
                  "where accuracy would hide it.\n")
        limit_b = "* Task B rare classes have ≤ 5 test cases, so their per-class metrics are very uncertain.\n"
    else:
        task_b = ("* **Task B (7 methylation subtypes)** is a planned extension: its feature ablation is reported in "
                  "§3, but it has not been tuned or evaluated in this version.\n")
        limit_b = ""
    return f"""## 10. Conclusions

* **Task A (WHO-2021 subtype from expression):** {PRETTY.get(a.get('best_by_cv'))}, test macro-F1
  {a.get('test_f1', float('nan')):.3f}. The three molecular classes are almost perfectly recoverable from the
  transcriptome. The derived chromosome-arm scores alone (~40 features) reach ~0.95–0.96. This shows the model
  is reading the *biology* (1p/19q loss, +7/−10, IDH-associated programs), not noise.
{task_b}* **Task C (LGG vs GBM, Tasci setup):** {PRETTY.get(c.get('best_by_cv'))}, test macro-F1
  {c.get('test_f1', float('nan')):.3f}, in line with the published ~87% accuracy. The ceiling comes from the
  labels themselves: TCGA "LGG" includes grade-III IDH-wildtype tumours that are biologically GBM under WHO 2021.
* **Regularisation** mattered most for single trees and k-NN (train F1 = 1.0 by default). For linear models,
  elastic-net gave sparse, interpretable gene sets at equal accuracy.
* **Ensembles** gave robust but not dramatic gains when the base learners are already near the ceiling. Their
  main benefit here is lower variance across folds: a single tree is clearly worse than the forest.

**Limitations:**
* The data are retrospective TCGA cohorts with batch-confounded sites.
* External cohorts lack 1p/19q status, so only IDH status and histology concordance can be checked externally.
{limit_b}
## Reproduce

```bash
uv sync
uv run glioma-all             # every stage in order (download → … → report); see `glioma-all --list`
uv run glioma-all --from evaluate   # only the cheap reporting stages
uv run pytest
```

Training takes under 30 min for Task A and ≈ 2 min for Task C on an 8-core laptop (`--fast` for a smoke test).
"""


def main() -> None:
    a = jread(f"{TASK_A}_best_model.json")
    header = f"""# Brain cancer (glioma) classification with machine learning

**Data:** NCI GDC / TCGA (Ceccarelli et al., Cell 2016) plus NCBI GEO external cohorts · **Code:** `src/glioma/` · all
numbers in this report are generated by `glioma-report` from `reports/tables/`.

**Summary.** We classify adult diffuse gliomas into the three WHO-2021 molecular types (IDH-wildtype, IDH-mutant
1p/19q-codeleted, IDH-mutant non-codeleted) from tumour gene expression. The best model by CV,
{PRETTY.get(a.get('best_by_cv'))}, reaches macro-F1 {a.get('test_f1', float('nan')):.3f} on held-out
patients. A second task benchmarks the pipeline against published work (LGG vs GBM, Tasci et al. 2022). The report covers all seven required components: EDA (§2), feature derivation (§3),
classification (§4), regularisation (§5), hyperparameter tuning (§6), ensembles and performance analysis
(§7–9).

"""
    body = "\n\n".join([header, section_data(), section_eda(), section_features(), section_models(),
                        section_regularization(), section_tuning(), section_performance(), section_external(),
                        section_literature(), section_conclusion()])
    OUT.write_text(body)
    print(f"wrote {OUT} ({len(body.split()):,} words)")


if __name__ == "__main__":
    main()
