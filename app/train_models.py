"""Re-fit the five tuned models from glioma_analysis.ipynb and save them for the demo app.

The split, log transform, 20% expression filter and the best hyperparameters are copied
from the notebook (sections 3 and 6), so the saved models reproduce the notebook's test scores.

Run from the project root:  python app/train_models.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
RAW = "data/raw/"
OUT = "app/artifacts/"
classes = ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"]

# 1. Data (notebook section 1)
expr = pd.read_csv(RAW + "LGG-GBM.gene_expression.normalized.txt", sep="\t", index_col=0)
meta = pd.read_excel(RAW + "TableS1.PatientData.20151020.v3.xlsx", sheet_name=0, header=1).set_index("Case")
labels = meta["IDH/codel subtype"].dropna()
ids = expr.index.intersection(labels.index)
X, y = expr.loc[ids], labels.loc[ids]

# 2. log2, split, filter, encode (notebook section 3)
X_log = np.log2(X + 1)
X_rest, X_te, y_rest, y_te = train_test_split(X_log, y, test_size=0.20, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_rest, y_rest, test_size=0.125, stratify=y_rest, random_state=SEED)
keep = (X_tr > 0).mean() >= 0.2
X_tr, X_va, X_te = X_tr.loc[:, keep], X_va.loc[:, keep], X_te.loc[:, keep]
code_of = {name: i for i, name in enumerate(classes)}
y_tr, y_va, y_te = y_tr.map(code_of), y_va.map(code_of), y_te.map(code_of)

# 3. Best hyperparameters found by the notebook's grid search (section 6 output)
models = {
    "Random Forest": Pipeline([("clf", RandomForestClassifier(
        n_estimators=200, max_features=0.2, min_samples_leaf=8, random_state=SEED, n_jobs=-1))]),
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(
        C=0.001, solver="lbfgs", max_iter=500, random_state=SEED))]),
    "XGBoost": Pipeline([("clf", XGBClassifier(
        objective="multi:softprob", tree_method="hist", subsample=0.8, eval_metric="mlogloss",
        n_estimators=300, max_depth=2, learning_rate=0.05, colsample_bytree=0.2, reg_lambda=1,
        n_jobs=-1, random_state=SEED))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=15, ccp_alpha=0, random_state=SEED))]),
    "KNN": Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=10, random_state=SEED)),
                     ("clf", KNeighborsClassifier(n_neighbors=3, weights="distance"))]),
}

scores = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    scores[name] = {"test_accuracy": accuracy_score(y_te, p), "test_macro_f1": f1_score(y_te, p, average="macro")}
    print(f"{name:20s} test acc {scores[name]['test_accuracy']:.3f}  macro-F1 {scores[name]['test_macro_f1']:.3f}")

# 4. Save models + the held-out test patients (log2 scale, filtered genes) for the demo
os.makedirs(OUT, exist_ok=True)
joblib.dump({"models": models, "genes": list(X_tr.columns), "classes": classes, "scores": scores}, OUT + "models.joblib", compress=3)
test = X_te.copy()
test.insert(0, "true_label", y_te.map(dict(enumerate(classes))))
test.insert(1, "grade", meta.loc[test.index, "Grade"])
test.to_csv(OUT + "test_patients.csv.gz", compression="gzip")
# A small raw-scale example file (RSEM values, as in the source data) to try the upload feature
expr.loc[X_te.index[:5]].to_csv(OUT + "example_upload_raw.csv")
print("saved to", OUT)
