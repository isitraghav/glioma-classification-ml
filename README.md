# Glioma (brain cancer) classification with machine learning

Predicts which of the three adult glioma types a tumour is, from its **gene expression** (RNA-seq). The labels
follow the WHO 2021 classification:

| Tumour type | Label |
|---|---|
| Glioblastoma | `IDHwt` |
| Astrocytoma | `IDHmut-non-codel` |
| Oligodendroglioma | `IDHmut-codel` |

A second task (lower-grade glioma vs glioblastoma) benchmarks the pipeline against Tasci et al. 2022.

| | Task A: 3 glioma types | Task C: LGG vs GBM |
|---|---|---|
| Patients (train / test) | 523 / 131 | 712 / 179 |
| Model selected by CV | Elastic-net logistic regression | Elastic-net logistic regression |
| Held-out test macro-F1 | **0.970** (95% CI 0.939–0.994) | **0.893** (accuracy 0.894 vs 0.876 published) |

**Models.**
- Logistic regression (elastic-net)
- Linear SVM
- RBF SVM
- k-NN
- Decision tree
- Random forest (bagging ensemble)
- Soft voting (LR + RBF-SVM + RF)

All share one leak-free pipeline: features → impute → scale → classifier.

**Data: official and research sources only, no Kaggle or GitHub.**
- NCI Genomic Data Commons: TCGA LGG+GBM, from the Ceccarelli et al. *Cell* 2016 publication page.
- GDC API: mutations and demographics.
- NCBI GEO: external cohorts GSE107850 and GSE16011.
- NCBI Gene.
- MSigDB.
- PubMed: literature scrape.

Every file is recorded with URL, date and MD5 in `data/raw/MANIFEST.csv`.

**Documents.**
- [`reports/report.md`](reports/report.md): full results, with all figures and tables, generated from `reports/tables/`.
- [`docs/PROJECT_END_TO_END.md`](docs/PROJECT_END_TO_END.md): how the dataset and models were built.
- [`docs/ML_Project_Synopsis.txt`](docs/ML_Project_Synopsis.txt): course synopsis.

## Run

Requires [`uv`](https://docs.astral.sh/uv/) and `aria2c` (`sudo apt install aria2`).

```bash
uv sync
uv run glioma-all --list      # show the stages
uv run glioma-all             # run everything: download (~354 MB) → build → EDA → train → evaluate → report
uv run glioma-all --from evaluate   # re-run only the cheap reporting stages
uv run pytest                 # leakage + sanity tests
```

Use the trained model on new samples. Rows are samples; columns are gene symbols, plus optional `age` and `sex`:

```bash
uv run glioma-predict --make-example demo.tsv     # 5 held-out TCGA patients
uv run glioma-predict --input demo.tsv --out predictions.csv
```

<details>
<summary>Individual stages</summary>

| Command | What it does |
|---|---|
| `glioma-download` | concurrent aria2c downloads + GDC API, provenance manifest |
| `glioma-literature` | PubMed scrape; published baselines extracted from abstracts |
| `glioma-build` | gene-name harmonisation, labels, mutation matrix, external cohorts |
| `glioma-eda` | EDA figures and tables (training split only) |
| `glioma-train` | feature ablation, default CV, random-search tuning, soft voting, repeated + nested CV (`--fast` for a smoke test) |
| `glioma-regularization` | regularisation study (Task A) |
| `glioma-tuning-plots` | hyperparameter-impact figures |
| `glioma-evaluate` | held-out test metrics, bootstrap CIs, statistical tests, SHAP, error analysis |
| `glioma-external` | external validation on the GEO cohorts |
| `glioma-report` | assemble `reports/report.md` |
| `glioma-predict` | predict glioma type for new samples |

</details>

## Layout

```
src/glioma/
  pipeline.py         glioma-all: runs every stage in order
  download.py         aria2c + GDC API downloads, provenance manifest
  literature.py       PubMed E-utilities scrape, baseline extraction
  build_dataset.py    raw -> data/processed/*.parquet
  data.py             task loading + frozen stratified train/test split
  features.py         GliomaFeatures (genes, chromosome-arm scores, pathway scores), TabularFeatures
  models.py           model zoo + search spaces
  train.py            CV / tuning / soft voting / nested CV
  regularization.py   regularisation study
  tuning_analysis.py  hyperparameter-impact plots
  evaluate.py         test-set performance analysis
  external.py         cross-platform external validation
  predict.py          glioma-predict CLI
  plots.py            shared plot style
  report.py           report generator
docs/                 end-to-end guide, course synopsis
reports/              figures/, tables/, report.md
models/<task>/        trained pipelines (*.joblib) + best_params.json   (git-ignored, re-creatable)
data/                 raw/, interim/, processed/                        (git-ignored, re-creatable)
tests/                leakage and sanity tests
```
