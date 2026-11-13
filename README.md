# Brain Cancer Type Classification from Tumour Gene Activity Using Machine Learning

## Problem
Adult diffuse gliomas are classified clinically by IDH mutation and 1p/19q codeletion, which needs molecular tests. This project asks whether a tumour's **gene expression (RNA-seq) alone** can predict the glioma type.

**Target (3 classes):** `IDH/codel subtype` from Ceccarelli 2016 Table S1

| Class | Patients |
|---|---|
| IDHmut-non-codel | 256 |
| IDHwt | 229 |
| IDHmut-codel | 169 |

## Data (`data/raw/`)
Both files come from the Ceccarelli 2016 TCGA LGG+GBM freeze (GDC portal):
- `LGG-GBM.gene_expression.normalized.txt`: RSEM-normalised expression, 667 patients x 20,330 genes. https://api.gdc.cancer.gov/data/0759ea74-25fa-4249-aabc-ae10617dda28
- `TableS1.PatientData.20151020.v3.xlsx` (sheet "S1A. TCGA discovery dataset"): per-patient labels. https://api.gdc.cancer.gov/data/a0886693-4c6e-40de-9cb1-8bfcb4bb30a7

The only data preparation is an inner join on the TCGA case ID after dropping unlabelled patients: **654 patients x 20,330 genes**. The raw files are never modified.

## Method (`glioma_analysis.ipynb`)
1. **EDA:** class balance, data quality, skew, p >> n, PCA, class vs grade.
2. **Features** (fitted on training data only): log2(x+1), drop genes expressed in under 20% of patients, standardise (LR, KNN), PCA (KNN only).
3. **Split:** 70% train / 10% validation / 20% test, stratified, seed 42 (457 / 66 / 131 patients). Train tunes, validation picks the model, test is used once at the end.
4. **Models and tuning:** Logistic Regression (L2), Decision Tree (depth, leaf size, pruning), KNN, Random Forest, XGBoost. Exhaustive search with 5-fold stratified CV, scored on macro-F1.
5. **Ensembles:** Random Forest (bagging) and XGBoost (boosting), compared against the single Decision Tree and each other.
6. **Evaluation:** accuracy, balanced accuracy, macro precision/recall/F1, ROC-AUC, PR-AUC, MCC, Cohen's kappa, log-loss, bootstrap 95% CIs, train-test gap, confusion matrices, top Random Forest genes.

See `QnA.md` for a full question-and-answer walkthrough of the project, its architecture and every design decision.

## Results
70/20/10 split, 131 test patients. The validation set picked **Random Forest** (validation macro-F1 1.000, but only 66 patients).

| Model | CV macro-F1 | Test accuracy | Test macro-F1 (95% CI) | ROC-AUC | Train-test gap |
|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.985 | 0.984 (0.957-1.000) | 0.992 | 0.015 |
| Random Forest | 0.983 | 0.969 | 0.971 (0.939-0.994) | 0.997 | 0.028 |
| XGBoost | 0.989 | 0.969 | 0.970 (0.938-0.993) | 0.999 | 0.031 |
| Decision Tree | 0.931 | 0.947 | 0.949 (0.908-0.984) | 0.956 | 0.012 |
| KNN | 0.953 | 0.931 | 0.934 (0.892-0.971) | 0.972 | 0.069 |

The top three models are statistically indistinguishable (overlapping CIs; one patient is about 0.8 points). Ensembles clearly beat a single tree (0.971 vs 0.949), but boosting did not beat bagging. Logistic Regression is the simplest model and scored highest on the test set, but it was not the validation pick.

## Limitations
- About 650 patients from one TCGA cohort, one split, no external validation, no batch correction.
- The 3-way split is relatively easy: IDH-mutant tumours have a strong expression signature, and IDHwt is mostly grade 4, so part of what is learned may be tumour severity.
- Several tuned values sit at the edge of their grid (LR `C`, tree depth, KNN neighbours), so the grids could be widened.
- Research exercise only, not a validated diagnostic.

## How to run
1. `pip install -r requirements.txt`
2. Open `glioma_analysis.ipynb` and run all cells (roughly 30-75 minutes depending on the CPU; the grid searches dominate, XGBoost most).
3. Demo app: `python app/train_models.py` (re-fits the five tuned models, about 1 minute), then `python -m streamlit run app/app.py`.
4. Report and slides: `python report/make_figures.py`, `python report/build_report.py` (needs MS Word for the PDF), `python report/build_slides.py`.

## Structure
```
README.md   requirements.txt   glioma_analysis.ipynb
data/raw/        source data
data/processed/  currently unused (the notebook saves nothing here)
report/          report.pdf (final report)
```
