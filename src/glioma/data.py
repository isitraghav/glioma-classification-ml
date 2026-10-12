"""Load task-specific (X, y) and the fixed stratified train/test split."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split

from glioma.build_dataset import TASK_A, TASK_B, TASK_C, TASKS
from glioma.config import PROCESSED, SEED, TEST_SIZE

CLASS_ORDER = {
    TASK_A: ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"],
    TASK_B: ["Classic-like", "Mesenchymal-like", "LGm6-GBM", "PA-like", "G-CIMP-low", "G-CIMP-high", "Codel"],
    TASK_C: ["LGG", "GBM"],
}


@dataclass
class TaskData:
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    classes: list[str]

    @property
    def description(self) -> str:
        return TASKS[self.name]


@lru_cache(maxsize=1)
def meta() -> pd.DataFrame:
    m = pd.read_parquet(PROCESSED / "tcga_meta.parquet")
    m["male"] = (m["gender"].str.lower() == "male").astype("float32")
    return m


@lru_cache(maxsize=1)
def expression() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "tcga_expr.parquet")


def load_xy(task: str) -> tuple[pd.DataFrame, pd.Series]:
    m = meta()
    if task in (TASK_A, TASK_B):
        e = expression()
        ids = e.index[e.index.isin(m.index[m[task].notna()])]
        X = e.loc[ids].join(m.loc[ids, ["age", "male"]])
    elif task == TASK_C:
        mut = pd.read_parquet(PROCESSED / "tcga_mutations.parquet")
        ids = mut.index[mut.index.isin(m.index[m[task].notna()])]
        X = mut.loc[ids].join(m.loc[ids, ["age", "male", "race"]])
    else:
        raise KeyError(task)
    y = m.loc[ids, task]
    return X, y


def load_task(task: str) -> TaskData:
    """Stratified 80/20 split; the split is frozen to disk the first time and re-used afterwards."""
    X, y = load_xy(task)
    split_file = PROCESSED / f"split_{task}.json"
    if split_file.exists():
        test_ids = json.loads(split_file.read_text())["test"]
    else:
        _, test_ids = train_test_split(list(X.index), test_size=TEST_SIZE, stratify=y, random_state=SEED)
        split_file.write_text(json.dumps({"seed": SEED, "test": sorted(test_ids)}, indent=1))
    test = X.index.isin(test_ids)
    return TaskData(task, X[~test], X[test], y[~test], y[test], CLASS_ORDER[task])
