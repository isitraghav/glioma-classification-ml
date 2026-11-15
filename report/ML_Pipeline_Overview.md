---
title: "Glioma Subtype Classification: Full ML Pipeline"
---

```
Problem → Data → Join → EDA → Transform → Split → Filter → Encode → Models (5)
→ Pipelines → Tuning (Grid + 5-fold CV) → Validation pick → Test (once)
→ Metrics + CIs → Over/underfit check → Error analysis → Gene interpretation
→ Saved models → Streamlit demo → Report & slides
```

## Stage 0: Problem definition
- **Task:** predict a glioma's molecular type from tumour gene expression alone.
- **Type:** supervised, 3-class classification.
- **Classes:** IDHwt (229), IDHmut-non-codel (256), IDHmut-codel (169).
- **Inputs:** gene expression only. Age, grade, IDH status and 1p/19q status are deliberately kept out, because IDH and 1p/19q define the label.
- **Main metric:** macro-F1, which weights the three classes equally despite the different class sizes.

## Stage 1: Data acquisition

| File | Contents |
|---|---|
| `LGG-GBM.gene_expression.normalized.txt` | RSEM-normalised RNA-seq, 667 patients × 20,330 genes |
| `TableS1.PatientData.20151020.v3.xlsx` | Ceccarelli 2016 labels (`IDH/codel subtype`) |

- **Source:** TCGA LGG+GBM, downloaded from the GDC portal.
- **Raw data:** the original files are never modified.

## Stage 2: Data integration
- Drop patients without a label.
- Inner-join expression and labels on the TCGA case ID.
- **Result:** 654 patients × 20,330 genes, with no missing values and no duplicates.

## Stage 3: Exploratory data analysis (EDA)
EDA uses all 654 patients, but only to describe the data. Nothing here feeds into the models.

| Check | Finding | Design decision it led to |
|---|---|---|
| Class balance | 256 / 229 / 169 | Stratified splits, macro-F1 |
| Dimensionality | 20,330 genes ≫ 654 patients (p ≫ n) | Strong regularisation, honest validation |
| Data quality | No missing values or duplicates | No imputation needed |
| Sparsity | 183 genes all-zero, 1,879 genes > 90% zero | Low-expression filter |
| Skew | Values from 0 to 884,605, median 519 | log2 transform |
| PCA | IDHwt separates on PC1; the two IDH-mutant classes overlap | Expect errors between the IDH-mutant classes |
| Class vs grade | IDHwt is mostly grade 4 | Noted as a confounder |

## Stage 4: Feature transformation
- Apply **log2(x + 1)** to every value.
- It's a fixed formula that learns nothing from the data, so it's safe to apply before the split.

## Stage 5: Data splitting
- **Order:** the 20% test set is taken first, then 12.5% of the remainder becomes validation.
- **Ratio:** 70 / 10 / 20, stratified, seed 42.

| Set | Patients | Role |
|---|---|---|
| Train | 457 | Fit models and tune them (cross-validation happens inside this set) |
| Validation | 66 | Pick the best of the 5 tuned models |
| Test | 131 | Final unbiased score, used once |

## Stage 6: Feature selection (low-expression filter)
- **Rule:** keep genes expressed (> 0) in at least 20% of the patients the filter is fitted on.
- **Implementation:** a custom scikit-learn transformer, `LowExpressionFilter`, placed as the **first step of every model's pipeline**. It is re-fitted inside every cross-validation fold, so held-out folds never influence which genes are kept.
- **Size:** when fitted on all 457 training patients, it keeps 17,963 of the 20,330 genes.
- **Why the filter is unsupervised:** it never looks at labels, so it cannot leak them.

## Stage 7: Label encoding
- IDHwt = 0, IDHmut-non-codel = 1, IDHmut-codel = 2.
- XGBoost requires integer labels.

## Stage 8: Model building
Each model is a scikit-learn `Pipeline`, so every learned step is fitted on training data only.

| # | Model | Family | Pipeline |
|---|---|---|---|
| 1 | Logistic Regression (L2) | Linear baseline | filter → StandardScaler → LR |
| 2 | Decision Tree | Single tree | filter → DT |
| 3 | KNN | Distance-based | filter → StandardScaler → PCA → KNN |
| 4 | Random Forest | Ensemble (bagging) | filter → RF |
| 5 | XGBoost | Ensemble (boosting) | filter → XGB |

**Regularisation in each model:**

- Logistic Regression: L2 penalty `C`.
- Decision Tree: depth limit, minimum leaf size, cost-complexity pruning.
- KNN: number of neighbours `k` and PCA.
- Random Forest: averaging many trees, `max_features`, minimum leaf size.
- XGBoost: learning rate, depth, `reg_lambda`, row and column subsampling.

## Stage 9: Hyperparameter tuning
- **Method:** `GridSearchCV`, an exhaustive search over every combination.
- **Cross-validation:** 5-fold stratified on the training set, shuffled, seed 42.
- **Score:** macro-F1.

| Model | Grid | Best settings | CV macro-F1 |
|---|---|---|---|
| LR | C ∈ {1e-4 … 1} | C = 0.001 | 0.983 |
| DT | depth, leaf, ccp_alpha | 3, 15, 0 | 0.931 |
| KNN | PCs, k, weights | 10, 3, distance | 0.953 |
| RF | trees, max_features, leaf | 500, 0.2, 1 | 0.985 |
| XGB | trees, depth, lr, colsample, lambda | 300, 2, 0.05, 0.2, 1 | 0.989 |

## Stage 10: Model selection
- Score each tuned model on the 66 validation patients:

| Model | Validation macro-F1 |
|---|---|
| **Logistic Regression** | **0.986 (selected)** |
| Random Forest | 0.986 |
| XGBoost | 0.986 |
| Decision Tree | 0.928 |
| KNN | 0.915 |

- Three models tied at 0.986, so the tie went to the simplest model, Logistic Regression.
- The choice was locked in before the test set was opened.

## Stage 11: Final evaluation on the test set
**Metrics:** accuracy, balanced accuracy, macro precision/recall/F1, ROC-AUC (one-vs-rest), PR-AUC, MCC, Cohen's kappa, log-loss.

| Model | Accuracy | Macro-F1 (95% CI) | ROC-AUC | Gap |
|---|---|---|---|---|
| **Logistic Regression** | 0.985 | 0.984 (0.957–1.000) | 0.992 | 0.015 |
| Random Forest | 0.977 | 0.978 (0.949–1.000) | 0.997 | 0.023 |
| XGBoost | 0.969 | 0.970 (0.938–0.993) | 0.999 | 0.031 |
| Decision Tree | 0.947 | 0.949 (0.908–0.984) | 0.956 | 0.012 |
| KNN | 0.931 | 0.934 (0.892–0.971) | 0.972 | 0.069 |

**Statistical check:** a bootstrap with 1,000 resamples of the test set gives 95% confidence intervals. The top three models overlap, so the data can't separate them.

## Stage 12: Overfitting and underfitting analysis
- **Overfit gap:** training accuracy minus test accuracy. KNN has the largest gap (0.069), so it overfits the most.
- **Cross-validation vs validation vs test:** for the top three models, cross-validation and test scores are within about 2 points, so the cross-validation estimates were reliable.
- **Underfitting:** the Decision Tree has the lowest training accuracy (0.958) and the lowest cross-validation score. At depth 3 it is too simple.

## Stage 13: Ensemble comparison
- **Bagging vs single tree:** Random Forest beat the Decision Tree, 0.978 vs 0.949. Bagging reduced variance.
- **Boosting vs bagging:** XGBoost was slightly below Random Forest, 0.970 vs 0.978.
- **Linear vs trees:** penalised Logistic Regression beat both ensembles (0.984), which is typical when there are far more genes than patients.

## Stage 14: Error analysis
- A confusion matrix for each of the five models.
- A per-class precision/recall report for the selected model.
- Logistic Regression got 129 of 131 test patients right. Both errors were IDHmut-non-codel predicted as IDHmut-codel.

## Stage 15: Model interpretation
- **Top 10 Random Forest genes:** RBP1, KIAA0495, PSMB2, FBXO42, FKBP9, MIER1, EMP3, FBXO17, LRRC41, CSDE1.
- **RBP1:** known to be switched off in IDH-mutant gliomas.
- **Chromosome locations:** 8 of the 10 lie on 1p (6 genes) or 19q (2 genes). This is consistent with the effects of 1p/19q codeletion, but it is a plausibility check, not proof.

## Stage 16: Model persistence
- `app/train_models.py` repeats the split and filter and re-fits the 5 tuned models.
- It saves the models, the gene list, the classes and the scores to `models.joblib`.
- It also saves the test patients and an example upload file for the demo.

## Stage 17: Deployment (demo app)
- `app/app.py` is a Streamlit app.
- **View 1:** pick a test patient and get the predicted type, the matching WHO type and each class's probability.
- **View 2:** upload a raw RSEM CSV; the app applies the same log transform and gene list.

## Stage 18: Documentation and reporting
- **Notebook:** `glioma_analysis.ipynb` runs everything end to end.
- **Report:** `report/build_report.py` produces the DOCX/PDF, and `make_figures.py` produces the charts.
- **Slides:** `report/build_slides.py` produces the PPTX.
- **Q&A and summary:** `QnA.md` and `README.md`.

## Stage 19: Limitations and future work
**Limitations:**

- One TCGA cohort, one split, no external validation, no batch correction.
- IDHwt is mostly grade 4, so the models may partly be learning tumour severity.
- Several tuned values sit at the edge of their grid.
- The validation set is small (66 patients).

**Next steps:**

- Validate on an independent cohort.
- Use wider grids and nested or repeated cross-validation.
- Try SVMs.
