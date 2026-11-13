# Project Q&A: Brain Cancer Type Classification from Tumour Gene Activity Using Machine Learning

One file that answers the questions anyone will ask about this project: what it is, how it is organised, what every part of the notebook does, why each choice was made, and where the limits are.

**Group:** 1 Raghav Yadav (240598) - 2 Saksham Lodha (240626) - 3 Rayna Manchanda (240601) - 4 Darshita Yadav (240981) - 5 Prerit Srivastava (240593)

> **Status note.** All numbers in this file come from the full run of the current notebook (70/20/10 split, five models incl. XGBoost, seed 42). Re-running gives the same split but small numeric differences can occur for the forest and XGBoost depending on hardware.

---

## Contents
1. Problem and motivation
2. Data
3. Folder structure and architecture
4. Notebook walkthrough, section by section
5. Design decisions and justifications
6. The models: how each works
7. Evaluation metrics
8. Overfitting, underfitting, bias-variance
9. Results
10. Limitations, and what the synopsis planned but we did not do
11. Syllabus mapping
12. Viva-style rapid-fire questions
13. How to run and troubleshooting

---

## 1. Problem and motivation

**Q1. What is the problem statement?**
Adult diffuse gliomas (brain tumours) are classified clinically by two molecular markers: **IDH mutation** and **1p/19q chromosome codeletion**. Finding them needs dedicated lab tests. The project asks whether a tumour's **gene expression (RNA-seq) alone** can predict the glioma type.

**Q2. What exactly are we predicting?**
A 3-class label, the `IDH/codel subtype` column of Table S1 from Ceccarelli et al. (2016):

| Class | Meaning | Approx. WHO 2021 type | Patients |
|---|---|---|---|
| IDHwt | no IDH mutation | glioblastoma | 229 |
| IDHmut-non-codel | IDH mutated, 1p/19q intact | astrocytoma | 256 |
| IDHmut-codel | IDH mutated, 1p/19q lost | oligodendroglioma | 169 |

The match to the WHO names is approximate (the IDHwt class also holds some lower-grade tumours).

**Q3. What is the input?**
About 20,000 gene expression values per patient (654 patients after joining expression with labels).

**Q4. What is the motivation?**
- The tumour type decides treatment and expected survival (from very aggressive glioblastoma to slow-growing oligodendroglioma).
- Since the 2021 WHO classification, doctors decide the type with genetic lab tests, which are expensive and not available in every hospital.
- A single measurement of gene activity already carries the "fingerprint" of each type. If a model can read it, one expression assay could stand in for several separate tests.
- With about 20,000 numbers per patient but only a few hundred patients, the task needs careful regularization and honest evaluation.

**Q5. Is this a regression or a classification problem, and why?**
Classification (supervised, 3 classes), because the target is a category with no natural order or distance between classes.

**Q6. What does "success" look like?**
High macro-F1 on patients the model never saw, a small gap between training and test scores, and a model that relies on plausible biology and not noise (checking this properly, e.g. gene locations, is future work).

---

## 2. Data

**Q7. Which datasets are used?**
Exactly two files, both from the Ceccarelli et al. (2016) TCGA lower-grade glioma + glioblastoma freeze, downloaded from the NCI Genomic Data Commons:

| File | Role | Size |
|---|---|---|
| `LGG-GBM.gene_expression.normalized.txt` | features: RSEM-normalised expression, 667 patients x 20,330 genes | 50 MB |
| `TableS1.PatientData.20151020.v3.xlsx` (sheet "S1A. TCGA discovery dataset") | labels (and grade, used only in EDA) | 0.4 MB |

**Q8. Why only these two?**
The original larger project had many more files (GEO cohorts, mutations, copy number, MSigDB gene sets). The expression matrix and Table S1 are the minimum needed for this task: one gives the features, the other the labels. Keeping two makes the project simple and easy to explain.

**Q9. What was changed in the raw data?**
Nothing on disk. In the notebook only:
1. Load expression with patients as rows and genes as columns.
2. Load Table S1 (header on its second row) keyed by the case ID.
3. Keep patients that have an `IDH/codel subtype` label (974 labelled).
4. Inner-join on case ID: only **654** patients have both expression and a label.
5. The log2 transform and the low-expression filter happen later, in the feature step.

**Q10. Why do only 654 of 974 labelled patients remain?**
The expression file has just 667 patients. The other labelled patients have no RNA-seq profile in this file. This is a limitation (possible selection bias).

**Q11. Are there missing values or duplicates?**
No missing expression values and no duplicate patients. 183 genes are zero in every patient and 1,879 are zero in over 90% of patients.

**Q12. Why is the value scale so strange?**
RSEM-normalised counts are extremely right-skewed: raw values reach 884,605 while the median is 519. Only a few genes are very active. That is why a log transform is needed. The data were also normalised to equal total expression per patient, so sequencing depth is not a confounder.

**Q13. Some gene columns are numbers, not names. Is that a problem?**
The file mixes some Entrez IDs with gene symbols as column names (for example `100133144` next to `A1BG`). It does not affect the models, because they use column positions. It only matters when reading gene names in the importance plot.

---

## 3. Folder structure and architecture

**Q14. What is the folder structure?**
```
final/
├── README.md                  # problem, data, method, results, how to run
├── QnA.md                     # this file
├── requirements.txt           # pandas, numpy, scikit-learn, matplotlib, openpyxl, jupyter, xgboost
├── glioma_analysis.ipynb      # the whole project: data, EDA, features, models, evaluation, conclusion
├── progress.txt               # status notes
├── data/
│   ├── raw/
│   │   ├── LGG-GBM.gene_expression.normalized.txt
│   │   └── TableS1.PatientData.20151020.v3.xlsx
│   └── processed/             # empty: the notebook does not save intermediate files
└── report/
    └── report.pdf             # final report
```
All code lives in one notebook and runs top to bottom in plain, flat cells (no functions or classes), so "Run All" works.

**Q15. What is the architecture of the pipeline?**
```
 Table S1 (labels) ───────────────┐
                                   ├─ inner join on case ID ─► 654 patients x 20,330 genes + label
 Expression matrix (features) ────┘
                                          │
                                  EDA (class balance, quality, skew, PCA, sparsity, top genes)
                                          │
                              log2(x + 1)  (fixed transform)
                                          │
              stratified split  70% train | 10% validation | 20% test
                                          │
                  low-expression filter learned on TRAIN only (genes expressed in >= 20%)
                                          │
        ┌───────────── 5 models, each a scikit-learn Pipeline ─────────────┐
        │ Logistic Regression   : StandardScaler -> LR (L2)                │
        │ Decision Tree         : tree                                     │
        │ KNN                   : StandardScaler -> PCA -> KNN             │
        │ Random Forest (bag)   : forest                                   │
        │ XGBoost (boost)       : gradient-boosted trees                   │
        └──────────────────────────────────────────────────────────────────┘
                                          │
        exhaustive hyperparameter search, 5-fold stratified CV on TRAIN, score = macro-F1
                                          │
        VALIDATION set  ──► pick the final model (best validation macro-F1)
                                          │
        TEST set (used once) ──► evaluation matrix, confusion matrices, bootstrap CIs
                                          │
                       gene importances, auto-generated summary, conclusion
```

**Q16. What does each of the three data splits do?**
- **Train (70%, 457 patients):** fits the models and, through 5-fold cross-validation inside it, tunes the hyperparameters.
- **Validation (10%, 66 patients):** used to choose between the five tuned models.
- **Test (20%, 131 patients):** untouched until the end, then used once for the final scores.

---

## 4. Notebook walkthrough, section by section

**Q17. What is in the notebook, in order?**

The notebook is split into small cells, each with a short markdown header. Longest code cell: 23 lines. No functions or classes.

| Section | What it does |
|---|---|
| Title, Problem Statement, Motivation, Literature Review | group members, the task, why it matters, the two references from the synopsis |
| Setup | imports, `SEED = 42`, the path, the class order list. Sets `OMP_NUM_THREADS=1` so parallel workers do not fight over CPU threads |
| 1.1 Load | reads the expression matrix and Table S1 |
| 1.2 Join | keeps patients with a label and expression, builds `X`, `y`, `grade` |
| 2.1 Class balance | bar chart of the three classes |
| 2.2 Data quality | missing values, duplicates, all-zero and mostly-zero genes, genes-to-patients ratio |
| 2.3 Distribution | histograms of raw vs log2 values (skew) |
| 2.4 PCA and grade | 2-D PCA coloured by class, and the class-by-grade table |
| 2.5 Gene sparsity | fraction of patients expressing each gene, with the 20% filter line |
| 2.6 Top genes | heatmap of the top 30 class-discriminating genes (descriptive only) |
| EDA findings | the conclusions drawn from the plots |
| 3. Features and split | explanation, then 3.1 log2 + 70/10/20 stratified split, 3.2 low-expression filter learned on train, 3.3 integer labels and class-mix check |
| 4. Base models | 4.1 Logistic Regression, 4.2 Decision Tree, 4.3 KNN: each cell adds a pipeline and its hyperparameter values to a `models` dictionary |
| 5. Ensemble models | 5.1 Random Forest (bagging), 5.2 XGBoost (boosting), added to the same dictionary |
| 6. Tuning | one loop runs the cross-validated search for every model and prints the best settings |
| 7.1 Validation | scores each tuned model on the validation set and selects one |
| 7.2 Test predictions | predictions and probabilities of every model on the test set |
| 7.3 Evaluation matrix | all metrics for all models, plus training accuracy and the overfit gap |
| 7.4 Confusion matrices | one per model |
| 7.5 Per-class report | precision, recall and F1 for the selected model |
| 7.6 Bootstrap | 95% confidence intervals for test macro-F1 |
| 7.7 Gene importance | top 10 Random Forest genes |
| 7.8 Interpretation, 7.9 Summary | how to read the results, then an automatic printout of the key numbers |
| 8. Conclusion | expected outcome, limitations, next steps |

**Q18. What does the data-loading cell do?**
`pd.read_csv(..., sep="\t", index_col=0)` reads the expression matrix with case IDs as the index. `pd.read_excel(..., sheet_name=0, header=1)` reads Table S1 (the real header is the second row). It takes the `IDH/codel subtype` column, drops missing values, intersects the case IDs with the expression index, and slices both. `grade` is kept only for EDA.

**Q19. What does the EDA tell us?**
- Classes are moderately imbalanced (256 / 229 / 169), so splits are stratified and macro-F1 is used.
- 20,330 genes for 654 patients (p >> n), so overfitting is the main risk.
- The values are heavily skewed, so a log transform is needed.
- IDHwt separates clearly from the IDH-mutant tumours along the first principal component, while the two IDH-mutant classes overlap heavily.
- The top class-discriminating genes form a clear block: IDHwt is high in CHI3L1, PDPN and EMP3, and the two IDH-mutant classes differ mainly in a few genes.
- IDHwt is mostly grade 4 (135 of 220 graded), IDHmut-codel has no grade 4, so the classes partly track tumour severity.

**Q20. Why is the top-30-genes heatmap not feature selection?**
It uses an ANOVA F-score on the *whole* dataset only to describe the data. It is never used to choose features for the models, because that would leak test information. The cell says so.

**Q21. What happens in the split and filter cells (3.1 to 3.3)?**
1. `X_log = np.log2(X + 1)`.
2. First split: 20% test. Second split: from the remaining 80%, take 12.5% (which is 10% of the total) as validation. Both are stratified with `random_state=42`. Result: 457 / 66 / 131 patients.
3. The filter: `keep = (X_tr > 0).mean() >= 0.2` is computed from the **training** patients only, and the same columns are kept in validation and test. About 17,963 genes remain.
4. Labels are mapped to integers (IDHwt=0, IDHmut-non-codel=1, IDHmut-codel=2), because XGBoost accepts only integer labels, and all models use the same labels.

**Q22. How are the pipelines built?**
Each model is a scikit-learn `Pipeline`, so any scaler or PCA is fitted only on the training fold during cross-validation. For example KNN is `StandardScaler -> PCA -> KNeighborsClassifier`.

**Q23. How does the tuning loop work?**
`GridSearchCV(pipeline, params, cv=cv, scoring="f1_macro", n_jobs=-1)` tries every combination of the listed values, with 5-fold stratified CV on the training set, and keeps the setting with the best average macro-F1. The best fitted pipeline and its CV score are stored for each model.

**Q24. Why are there no functions or classes in the notebook?**
By request: plain top-to-bottom cells so the notebook can be run automatically with Run All. Loops and dictionaries are used instead.

**Q25. What is in the evaluation matrix?**
For every model on the test set: accuracy, balanced accuracy, macro precision, macro recall, macro-F1, ROC-AUC (one-vs-rest), PR-AUC, MCC, Cohen's kappa and log-loss.

---

## 5. Design decisions and justifications

**Q26. Why log2(x + 1)?**
Raw RSEM values span many orders of magnitude, so a few very active genes would dominate any distance- or weight-based model. The log compresses the range and stabilises variance. The `+ 1` keeps zeros defined. It is a fixed transform that learns nothing from the data, so it can safely be applied before the split.

**Q27. Why filter genes expressed in under 20% of patients?**
Genes that are zero for most patients carry almost no information but add noise dimensions in a p >> n problem. The EDA found 183 all-zero and 1,879 mostly-zero genes. The 20% threshold is a judgement call and was not tuned. The filter is learned on the training patients only so nothing leaks from validation or test.

**Q28. Why not select features with ANOVA or similar?**
An earlier version used an ANOVA top-k step. It was replaced by the explicit, unsupervised low-expression filter, which is simpler to explain and cannot leak labels. The models cope with the remaining ~18,000 genes through regularization (L2, tree limits, shrinkage, column subsampling).

**Q29. Why standardise only some models?**
Logistic regression (a penalised linear model) and KNN (distance-based) depend on feature scale. Trees, Random Forest and XGBoost split on thresholds, so scaling does not change them.

**Q30. Why PCA for KNN only?**
In ~18,000 dimensions all patients look about equally far apart, so nearest neighbours lose meaning. PCA compresses genes into a few components first (10 or 30, tuned). The other models handle high dimensions on their own.

**Q31. Why a 70/20/10 split?**
It gives a large training set, a large test set for a trustworthy final score, and a small validation set for choosing between models. This also fixes a flaw in the earlier version, where the "best" model was picked using the test set. With only 654 patients, 10% validation is just 66 patients, so the validation choice is noisy (see limitations).

**Q32. Why stratified splits and stratified CV?**
So every subset keeps the same class mix. Otherwise a split could by chance contain few IDHmut-codel patients (the smallest class) and give misleading scores.

**Q33. Why cross-validation for tuning, not just the validation set?**
5-fold CV uses the whole training set for both fitting and scoring, which gives a steadier estimate than one small validation split. The small validation set is kept only for the final choice between models.

**Q34. Why macro-F1 as the main metric?**
F1 balances precision and recall; macro-averaging weights the three classes equally. The classes are unequal in size (169 to 256), so plain accuracy could hide poor performance on the smallest class.

**Q35. Why exhaustive grid search and not random search?**
It was requested. With these small grids (5 to 36 combinations per model) an exhaustive search is affordable and fully reproducible. The grid sizes are: Logistic Regression 5, Decision Tree 36, KNN 16, Random Forest 12, XGBoost 32 combinations, each fitted 5 times.

**Q36. Why was logistic regression restricted to L2 (lbfgs)?**
L1 with the `saga` solver was extremely slow on ~18,000 standardised genes: a full run did not finish logistic regression in over 15 minutes. `lbfgs` with L2 trains in well under a minute. L2 still gives regularization (shrinkage). The cost is that L1's automatic gene selection is gone.

**Q37. Why these particular models?**
- **Logistic Regression:** strong, simple baseline for expression data, gives signed gene weights.
- **Decision Tree:** interpretable, and the natural "weak" model to show what ensembles improve.
- **KNN:** non-parametric, distance-based contrast to the others.
- **Random Forest:** bagging ensemble, reduces tree variance.
- **XGBoost:** boosting ensemble, a different ensemble family from bagging.
The first four were requested, XGBoost was added as a second ensemble technique.

**Q38. Why add XGBoost?**
The assignment needs at least one ensemble method. Random Forest covers bagging. XGBoost adds boosting, the other main family, so the notebook can compare bagging vs boosting and the syllabus topic "Bagging and Boosting" is covered.

**Q39. Why a separate validation step to pick the final model?**
Choosing the winner on the test set lets test information influence the choice, which makes the final score slightly optimistic. With a validation set, the test set stays untouched until the end.

**Q40. Why the bootstrap confidence intervals?**
A test score from 131 patients has real uncertainty: one patient is about 0.8 points of accuracy. Resampling the test patients 1,000 times gives a 95% interval for macro-F1, so you can judge whether two models genuinely differ. If their intervals overlap heavily, the data cannot separate them.

**Q41. Why is the random seed fixed at 42?**
So splits, tree-based models and CV folds are reproducible. Results will vary slightly with another seed.

**Q42. Why `n_jobs=1` inside the Random Forest and XGBoost, but `n_jobs=-1` in the search?**
Parallelism is taken at one level only (the search runs many fits at once). Parallel models inside parallel workers would oversubscribe the CPU.

---

## 6. The models: how each works

**Q43. How does Logistic Regression work here?**
It gives each gene a weight and combines them into a score per class, turned into probabilities with softmax. It is fitted by minimising log-loss plus an **L2 penalty** that shrinks weights toward zero. `C` is the *inverse* of the penalty strength (small C = stronger shrinkage). Tuned: `C` in {1e-4, 1e-3, 1e-2, 0.1, 1}. Strength: simple, fast, well suited when many genes shift consistently. Weakness: assumes a roughly linear boundary.

**Q44. How does a Decision Tree work, and how is it regularised?**
It asks a chain of yes/no questions about single genes (`gene > threshold?`) and picks the split that best separates the classes (gini impurity). Left alone it grows until it memorises the training data. Regularization: `max_depth`, `min_samples_leaf` and cost-complexity pruning `ccp_alpha`. Tuned: depth {3,5,8,None}, leaf {1,5,15}, alpha {0,0.005,0.02}. Weakness: high variance.

**Q45. How does KNN work?**
For a new patient it finds the `k` most similar training patients (in PCA space) and takes a majority vote, optionally weighted by distance. Small `k` is flexible but noisy (overfits), large `k` is smooth but can blur classes. Tuned: PCA components {10,30}, `n_neighbors` {3,7,15,25}, weights {uniform, distance}.

**Q46. How does Random Forest work?**
It grows many trees, each on a bootstrap resample of the patients and using a random subset of genes at every split, then averages their votes. Averaging de-correlated trees reduces variance without increasing bias much. Regularization comes from the averaging, `max_features` and `min_samples_leaf`. Tuned: trees {200,500}, `max_features` {sqrt, 0.2}, leaf {1,3,8}. It is **bagging**: trees are built independently.

**Q47. How does XGBoost work?**
It is **gradient boosting**: trees are built one after another, and each new small tree is fitted to the errors (the gradient of the log-loss) of the model so far. The predictions of all trees are added up, scaled by the learning rate. This mainly reduces *bias*. Regularization: shallow trees (`max_depth`), small `learning_rate`, L2 penalty `reg_lambda`, column subsampling `colsample_bytree` and row subsampling (`subsample=0.8`). Tuned: trees {100,300}, depth {2,4}, learning rate {0.05,0.2}, colsample {0.2,0.5}, lambda {1,10}. Fixed: `multi:softprob`, `hist` tree method, `mlogloss`.

**Q48. Bagging vs boosting?**
| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Trees built | independently, in parallel | sequentially, each fixing the last |
| Main effect | lowers variance | lowers bias |
| Tree depth | deep, even fully grown | shallow |
| Overfitting risk | low | higher if too many trees or a large learning rate |

**Q49. Why can a single Decision Tree be worse than a forest?**
One tree is unstable: small changes in the training patients change the splits. Averaging many different trees cancels those individual mistakes.

---

## 7. Evaluation metrics

**Q50. What does each metric mean?**
- **Accuracy:** share of patients classified correctly. Misleading if classes are unequal.
- **Balanced accuracy:** average recall per class.
- **Precision (per class):** of the patients predicted as this class, how many truly are.
- **Recall (per class):** of the patients truly in this class, how many were found.
- **F1:** harmonic mean of precision and recall. **Macro-F1** averages the three classes equally: the primary metric.
- **ROC-AUC (one-vs-rest):** how well the probabilities rank true members above non-members, averaged over classes. Threshold-free.
- **PR-AUC:** area under the precision-recall curve (average precision), more sensitive when classes are small.
- **MCC (Matthews correlation):** one number from the whole confusion matrix, from -1 to 1, robust to class imbalance.
- **Cohen's kappa:** agreement beyond chance.
- **Log-loss:** penalises confident wrong probabilities. Lower is better, and it also measures probability quality.
- **Confusion matrix:** rows are true classes, columns are predictions; off-diagonal cells are the mistakes.
- **Bootstrap 95% CI:** the range of macro-F1 you would expect if the test set were drawn again.

**Q51. Which metric picks the winner?**
Validation macro-F1 selects the final model. Test macro-F1 (with its CI) reports how good it is.

---

## 8. Overfitting, underfitting, bias-variance

**Q52. How do we detect overfitting?**
By comparing scores on data the model trained on with data it did not see: the **train-test accuracy gap** and the gap between cross-validated and test scores. A model that memorises the training patients scores near 100% there but worse on new ones.

**Q53. How do we detect underfitting?**
Low scores on both training and held-out data. Strong regularization (very small `C`, very shallow trees) can cause it. In our run all models scored well on unseen patients (test macro-F1 0.93-0.98), so none clearly underfit.

**Q54. What controls overfitting in this project?**
The low-expression filter, L2 penalty, tree depth and pruning, KNN neighbourhood size and PCA, forest averaging, XGBoost's learning rate / depth / lambda / subsampling, and honest evaluation (CV, validation and test sets).

**Q55. What is the bias-variance trade-off here?**
A flexible model (deep tree, KNN with k=3) has low bias but high variance: it fits the training noise. A rigid model (very strong penalty) has high bias. Bagging reduces variance; boosting reduces bias. Tuning on cross-validated macro-F1 looks for the balance.

**Q56. Does 100% training accuracy always mean overfitting?**
No. A random forest grows trees to fit the training data, so near-100% training accuracy is normal for it. The train-test gap is the more meaningful number.

---

## 9. Results

**Q57. What are the results?**
Split: 457 train / 66 validation / 131 test patients. Hyperparameters were tuned by 5-fold CV on the training set. The validation set picked **Random Forest** (validation macro-F1 1.000; Logistic Regression and XGBoost 0.986, Decision Tree 0.928, KNN 0.915). Test results:

| Model | CV macro-F1 | Test acc | Test macro-F1 (95% CI) | ROC-AUC | MCC | Log-loss | Gap |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.985 | 0.984 (0.957-1.000) | 0.992 | 0.977 | 0.098 | 0.015 |
| Random Forest | 0.983 | 0.969 | 0.971 (0.939-0.994) | 0.997 | 0.954 | 0.135 | 0.028 |
| XGBoost | 0.989 | 0.969 | 0.970 (0.938-0.993) | 0.999 | 0.954 | 0.077 | 0.031 |
| Decision Tree | 0.931 | 0.947 | 0.949 (0.908-0.984) | 0.956 | 0.919 | 1.960 | 0.012 |
| KNN | 0.953 | 0.931 | 0.934 (0.892-0.971) | 0.972 | 0.896 | 1.440 | 0.069 |

Best settings: LR `C`=0.001; Decision Tree depth 3, leaf 15, no pruning; KNN 10 PCA components, k=3, distance weights; Random Forest 200 trees, `max_features`=0.2, leaf 8; XGBoost 300 trees, depth 2, learning rate 0.05, colsample 0.2, lambda 1.

Errors out of 131 test patients: Logistic Regression 2, Random Forest 4, XGBoost 4, Decision Tree 7, KNN 9. The selected Random Forest's errors: 2 IDHwt predicted as non-codel, 1 non-codel predicted as IDHwt, 1 non-codel predicted as codel. Every model except KNN classified all 34 IDHmut-codel test patients correctly.

**Q58. What do the results tell us?**
- Logistic Regression, Random Forest and XGBoost are tied within noise (overlapping CIs; one patient is about 0.8 points).
- **Bagging vs single tree:** Random Forest beats the Decision Tree (test macro-F1 0.971 vs 0.949; CV 0.983 vs 0.931).
- **Boosting vs bagging:** XGBoost had the best CV score but the same test score as Random Forest (0.970 vs 0.971). No measurable gain from boosting here.
- KNN overfits most (gap 0.069) and is weakest.
- The validation set (66 patients) picked Random Forest, but Logistic Regression scored higher on test. This shows how noisy a 66-patient validation choice is.
- Log-loss separates models more than accuracy: the tree and KNN give overconfident probabilities.
- An earlier 80/20 run (no validation set, no XGBoost) gave a similar picture (RF and LR tied at about 0.98), so the conclusion does not depend on the exact split.

**Q59. If two models tie, how do we choose?**
Honestly: say they tie. Statistics cannot separate them at this sample size. A reasonable tie-breaker is parsimony: logistic regression is simpler, faster and more interpretable than a forest, and here it also had the highest test macro-F1. Our recommendation is Logistic Regression as the main model, with Random Forest (the validation-selected model) as the ensemble.

**Q60. Why are the scores so high? Is it too good to be true?**
IDH-mutant tumours have a strong, well-known expression signature, so the 3-way split is relatively easy. The test set is small and from one cohort, so the scores should not be read as proof of generalisation to other hospitals. Part of what is learned may also be tumour grade, since IDHwt is mostly grade 4.

**Q61. What do the top genes mean?**
The Random Forest importance plot lists, in order: RBP1, KIAA0495, PSMB2, MIER1, FBXO42, CSDE1, FBXO17, TMEM167B, FKBP9, EMP3. Six of them (RBP1, KIAA0495, FBXO17, TMEM167B, FKBP9, EMP3) also appear in the ANOVA top-30 heatmap from the EDA, so the forest and the univariate test agree. We have **not** checked where these genes lie on the chromosomes, so we do not claim the model has learned the 1p/19q signal. Importances do not prove biological mechanism, and they are biased toward high-variance genes and split credit among correlated genes.

---

## 10. Limitations, and what the synopsis planned but we did not do

**Q62. What are the limitations?**
- About 650 patients from a single TCGA cohort, one split, no external validation.
- 320 of the 974 labelled patients have no expression profile (possible selection bias).
- No batch or tissue-source correction.
- The 10% validation set has only 66 patients, so the model choice is noisy.
- IDHwt is mostly grade 4, so part of the signal may be severity.
- Hyperparameter ranges are small; winners at the edge of a range may not be the true optimum.
- The 20% filter threshold was not tuned.
- Research exercise only, not a validated diagnostic tool.

**Q63. What did the project synopsis propose that is not in the notebook?**
- Models: linear SVM, RBF SVM, soft voting.
- Features: chromosome-arm scores (39), pathway scores from Hallmark and Verhaak gene sets (54), age and sex.
- Methods: random search with nested cross-validation, SHAP explanations, calibration, MCC/kappa significance tests (corrected t-test, McNemar), an 80/20 split, gene-name mapping through NCBI Gene.
- Class names: the synopsis calls the classes glioblastoma / astrocytoma / oligodendroglioma; the notebook uses the molecular names (IDHwt / IDHmut-non-codel / IDHmut-codel).
The project was deliberately simplified to two datasets and a flat notebook. These items are listed as next steps.

**Q64. What would improve the project most?**
An independent test cohort; wider tuning ranges; repeated cross-validation; adding SVMs; adding the chromosome-arm and pathway features from the synopsis.

---

## 11. Syllabus mapping (ML course CSE2032)

| Course topic | Where it appears |
|---|---|
| Types of learning, classification | supervised 3-class classification |
| Sampling | stratified splitting |
| Under/overfitting, bias-variance | section 8, train-test gap |
| Regularization | L2 in logistic regression, tree pruning, XGBoost penalties |
| Logistic regression, decision tree, KNN | models 1 to 3 |
| Ensemble: decision forest, bagging, boosting | Random Forest (bagging), XGBoost (boosting) |
| Dimension reduction with PCA | EDA plot and the KNN pipeline |
| Model validation: accuracy, confusion matrix, precision, recall, ROC, cross-validation, bootstrap, hyper-parameter tuning | sections 6 and 7.1 to 7.6 |
| Not covered | SVM, Lasso, from-scratch implementations, clustering, Bayesian methods, neural networks, reinforcement learning |

---

## 12. Viva-style rapid-fire questions

**Q65. Why not use accuracy only?** Classes are unequal; macro-F1 and balanced accuracy weigh each class equally.
**Q66. What is p >> n and why is it dangerous?** More features (20,330) than samples (654). Models can fit noise perfectly, so they overfit.
**Q67. What is data leakage and how did you avoid it?** Information from test data influencing training. We split first, learned the gene filter on training patients only, and put scalers/PCA inside pipelines so they are fitted per fold.
**Q68. Why not do the log transform after splitting?** It is a fixed formula with no learned parameters, so it cannot leak.
**Q69. What is the difference between validation and test sets?** Validation chooses the model; test is used once for the final unbiased score.
**Q70. Why is KNN usually worst here?** Distances in high dimensions are weak, and its best `k` is very small, which makes it noisy and prone to overfitting.
**Q71. What does L2 regularization do?** Adds the sum of squared weights to the loss, shrinking weights and so reducing variance.
**Q72. L1 vs L2?** L1 can set weights exactly to zero (feature selection); L2 shrinks all weights but keeps them non-zero.
**Q73. What does `C` mean?** Inverse regularization strength: smaller `C` means a stronger penalty.
**Q74. What is gini impurity?** A measure of how mixed the classes are in a node. 0 means pure.
**Q75. What is out-of-bag / bootstrap sampling?** Drawing patients with replacement. About a third of patients are left out of each tree's sample.
**Q76. Why does XGBoost need integer labels?** Its scikit-learn wrapper rejects strings, so classes are encoded 0/1/2 and the names are restored for plots.
**Q77. What is PCA?** A linear projection onto directions of maximum variance. PC1 here separates IDHwt from IDH-mutant tumours.
**Q78. What is stratification?** Keeping the same class proportions in each split or fold.
**Q79. What is a confusion matrix?** A table of true vs predicted classes; the diagonal holds correct predictions.
**Q80. What is ROC-AUC?** The probability that a random positive is ranked above a random negative. 0.5 is chance, 1.0 is perfect.
**Q81. How would you improve it?** External validation, wider tuning ranges, repeated or nested CV, SVMs, biologically derived features.
**Q82. Can this be used clinically?** No. It is a research exercise on one retrospective cohort and is not validated for diagnosis.

---

## 13. How to run and troubleshooting

**Q83. How do I run the project?**
1. `pip install -r requirements.txt` (or use the existing virtual environment, which already has the packages).
2. Open `glioma_analysis.ipynb` and **Run All**. Run it from the `final/` folder, because the data path is `data/raw/`.

**Q84. How long does it take?**
The cross-validated searches dominate. The full run took about 76 minutes on a 16-core CPU (the four non-XGBoost searches took about 9-19 minutes; the XGBoost search with 160 fits is the slowest part). The first search (logistic regression) should finish in under a minute.

**Q85. Why is the GPU not used?**
scikit-learn runs on the CPU only, and at this data size (a few hundred patients) a GPU would help little. XGBoost can use a GPU but it is not set up here.

**Q86. Problems met while building, and the fixes**
- Logistic regression with L1 and the `saga` solver never finished: switched to L2 with `lbfgs`.
- XGBoost rejected string labels: labels are encoded as integers 0/1/2 for all models.
- Silent notebook failures came from a bug in the build script, not from the models.
- Parallel runs oversubscribed the CPU: worker math threads are pinned to one with `OMP_NUM_THREADS=1`.

**Q87. Where are the processed datasets?**
There are none. The joined and filtered tables exist only in memory while the notebook runs, and `data/processed/` is empty. A cell can be added to save them as CSV or parquet if needed.
