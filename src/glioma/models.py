"""Model zoo: pipelines, sensible defaults, and hyperparameter search spaces.

Every model is a full Pipeline (features -> impute -> scale -> [reduce] -> model), so feature
derivation and scaling are re-fit inside each CV fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from joblib import Memory
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from glioma.build_dataset import TASK_C
from glioma.config import INTERIM, SEED
from glioma.features import GliomaFeatures, TabularFeatures

# Caches fitted feature/scaling steps so hyperparameter candidates that share them reuse the work.
# Cleared at the end of each training run (it is only useful within a run and grows to >1 GB).
CACHE = Memory(INTERIM / "pipeline_cache", verbose=0)

CLASSICAL = ["logreg", "svm_linear", "svm_rbf", "knn", "tree"]
ENSEMBLES = ["random_forest", "soft_voting"]
VOTING_MEMBERS = ["logreg", "svm_rbf", "random_forest"]


@dataclass
class Spec:
    estimator: object
    space: dict = field(default_factory=dict)        # RandomizedSearchCV distributions (pipeline names)
    n_iter: int = 25
    reduce: bool = False                              # model benefits from a PCA step
    family: str = "classical"
    explanation: str = ""


def svm(kernel: str, **kw):
    # SVC(probability=True) is deprecated in sklearn 1.9 -> Platt scaling via CalibratedClassifierCV
    return CalibratedClassifierCV(SVC(kernel=kernel, class_weight="balanced", random_state=SEED, **kw),
                                  ensemble=False, cv=3, method="sigmoid")


def specs(task: str) -> dict[str, Spec]:
    expr = task != TASK_C
    k = {"feat__k_genes": [500, 1000, 2000, 4000]} if expr else {}
    r = {"reduce__n_components": [5, 10, 20, 30, 50]} if expr else {}
    return {
        "logreg": Spec(
            LogisticRegression(solver="saga", l1_ratio=0.5, C=0.1, max_iter=3000, tol=1e-3,
                               class_weight="balanced", random_state=SEED),
            {**k, "model__C": loguniform(1e-3, 10), "model__l1_ratio": uniform(0, 1)},
            explanation="Multinomial logistic regression: linear log-odds per class, softmax output. "
                        "Elastic-net (L1+L2) penalty -> sparse, stable gene signatures when p >> n."),
        "svm_linear": Spec(
            svm("linear", C=0.01),
            {**k, "model__estimator__C": loguniform(1e-5, 1)},
            explanation="Maximum-margin linear separator; C trades margin width against training errors."),
        "svm_rbf": Spec(
            svm("rbf", C=10, gamma="scale"),
            {**k, "model__estimator__C": loguniform(1e-1, 1e3), "model__estimator__gamma": loguniform(1e-5, 1e-1)},
            explanation="Kernel SVM: RBF kernel gives a non-linear boundary; gamma sets kernel width."),
        "knn": Spec(
            KNeighborsClassifier(n_neighbors=10, weights="distance"),
            {**k, **r, "model__n_neighbors": randint(3, 40),
             "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
            reduce=expr,
            explanation="Instance-based: majority vote of nearest training samples (after PCA, because "
                        "distances degrade in 2,000-D space)."),
        "tree": Spec(
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {**k, "model__max_depth": randint(2, 15), "model__min_samples_leaf": randint(1, 25),
             "model__ccp_alpha": loguniform(1e-4, 5e-2), "model__criterion": ["gini", "entropy"]},
            explanation="Greedy axis-aligned splits that maximise impurity reduction; interpretable but "
                        "high variance — depth / leaf size / cost-complexity pruning regularise it."),
        "random_forest": Spec(
            RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED,
                                   n_jobs=1),
            {**k, "model__max_features": ["sqrt", "log2", 0.05, 0.1, 0.2],
             "model__min_samples_leaf": randint(1, 8), "model__max_depth": [None, 6, 10, 20]},
            n_iter=15, family="ensemble",
            explanation="Bagging + random feature subsets per split -> de-correlated trees, lower variance."),
    }


def make_pipeline(task: str, name: str, model=None, reduce: bool | None = None) -> Pipeline:
    spec = specs(task).get(name)
    model = model if model is not None else spec.estimator
    reduce = spec.reduce if reduce is None else reduce
    feat = GliomaFeatures() if task != TASK_C else TabularFeatures()
    steps = [("feat", feat), ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    steps.append(("reduce", PCA(n_components=20, random_state=SEED) if reduce else "passthrough"))
    steps.append(("model", model))
    return Pipeline(steps, memory=CACHE)


def soft_voting(members: dict[str, Pipeline]) -> VotingClassifier:
    return VotingClassifier(list(members.items()), voting="soft", n_jobs=1)

