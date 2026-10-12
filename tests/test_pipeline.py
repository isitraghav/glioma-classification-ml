"""Leakage and sanity tests (run with `uv run pytest`). Require `glioma-build` to have been run."""
import numpy as np
import pandas as pd
import pytest

from glioma.build_dataset import TASK_A, TASK_B, TASK_C
from glioma.config import PROCESSED
from glioma.data import load_task
from glioma.features import GliomaFeatures, SampleRankNormalizer, TabularFeatures
from glioma.models import make_pipeline

pytestmark = pytest.mark.skipif(not (PROCESSED / "tcga_expr.parquet").exists(), reason="run glioma-build first")

# metadata columns that define (or are derived from) the labels
LABEL_DEFINING = {"idh_status", "codel_1p19q", "chr7gain_chr10loss", TASK_A, TASK_B, TASK_C, "study",
                  "histology", "who_grade", "verhaak_subtype"}


@pytest.mark.parametrize("task", [TASK_A, TASK_B])
def test_no_label_defining_inputs(task):
    d = load_task(task)
    assert not LABEL_DEFINING & set(d.X_train.columns)
    # Columns named IDH1/IDH2 are the genes' *expression* (continuous, legitimate). Mutation calls
    # (binary int8 flags) and copy-number values must never be inputs of the expression tasks.
    genes = d.X_train.drop(columns=["age", "male"])
    assert all(t.kind == "f" for t in genes.dtypes)
    assert genes["IDH1"].nunique() > 50


@pytest.mark.parametrize("task", [TASK_A, TASK_B, TASK_C])
def test_split_is_disjoint_stratified_and_frozen(task):
    d = load_task(task)
    assert not set(d.X_train.index) & set(d.X_test.index)
    assert abs(len(d.X_test) / (len(d.X_test) + len(d.X_train)) - 0.2) < 0.01
    p_tr = d.y_train.value_counts(normalize=True)
    p_te = d.y_test.value_counts(normalize=True)
    assert (p_tr - p_te).abs().max() < 0.05
    assert load_task(task).X_test.index.equals(d.X_test.index)


def test_features_fit_on_train_only():
    d = load_task(TASK_A)
    f = GliomaFeatures(k_genes=500).fit(d.X_train)
    mu_before = f.mu_.copy()
    f.transform(d.X_test)
    np.testing.assert_array_equal(mu_before, f.mu_)
    g = GliomaFeatures(k_genes=500).fit(pd.concat([d.X_train, d.X_test]))
    assert not np.allclose(f.mu_, g.mu_)  # test samples would change the statistics


def test_feature_shapes_and_names():
    d = load_task(TASK_A)
    f = GliomaFeatures(k_genes=1000).fit(d.X_train)
    Z = f.transform(d.X_test)
    assert Z.shape == (len(d.X_test), len(f.get_feature_names_out()))
    assert sum(n.startswith("gene:") for n in Z.columns) == 1000
    assert {"arm:1p", "arm:19q", "arm:7q", "arm:10q"} <= set(Z.columns)
    assert np.isfinite(Z.drop(columns=["age"]).to_numpy()).all()


def test_arm_scores_separate_codeletion():
    d = load_task(TASK_A)
    Z = GliomaFeatures(use_genes=False, use_sets=False).fit(d.X_train).transform(d.X_train)
    codel = d.y_train == "IDHmut-codel"
    assert Z.loc[codel, "arm:1p"].mean() < Z.loc[~codel, "arm:1p"].mean() - 0.3
    assert Z.loc[codel, "arm:19q"].mean() < Z.loc[~codel, "arm:19q"].mean() - 0.3


def test_rank_normalizer_is_stateless_and_bounded():
    d = load_task(TASK_A)
    R = SampleRankNormalizer().fit_transform(d.X_test.iloc[:5])
    genes = [c for c in R.columns if c not in ("age", "male")]
    assert R[genes].to_numpy().max() <= 1 and R[genes].to_numpy().min() > 0
    pd.testing.assert_series_equal(R["age"], d.X_test.iloc[:5]["age"])


def test_task_c_features():
    d = load_task(TASK_C)
    f = TabularFeatures().fit(d.X_train)
    Z = f.transform(d.X_test)
    assert {"idh_any", "astro_sig", "oligo_sig", "gbm_sig"} <= set(Z.columns)
    assert Z.shape[0] == len(d.X_test)


def test_pipeline_fits_and_predicts():
    d = load_task(TASK_C)
    pipe = make_pipeline(TASK_C, "logreg").fit(d.X_train, d.y_train)
    proba = pipe.predict_proba(d.X_test)
    assert proba.shape == (len(d.X_test), 2)
    assert np.allclose(proba.sum(1), 1)
