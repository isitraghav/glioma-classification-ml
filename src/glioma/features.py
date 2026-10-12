"""Feature derivation as scikit-learn transformers.

Every transformer learns its statistics in ``fit`` only, so inside a Pipeline the statistics
come from the training fold alone (no test-fold leakage).

Expression features (Tasks A/B)
-------------------------------
* ``genes``    top-k most variable genes (k is a hyperparameter)
* ``arms``     chromosome-arm expression scores: mean gene z-score per autosomal arm, re-centred on
               the sample's genome-wide mean. Copy-number loss lowers the expression of every gene on
               the arm, so 1p/19q scores act as an expression proxy for 1p/19q codeletion, and the
               7 vs 10 scores as a proxy for the +7/-10 glioblastoma signature.
* ``sets``     single-sample pathway scores (mean z) for the 50 MSigDB Hallmarks and the four
               Verhaak 2010 GBM transcriptional subtypes.
* ``clinical`` age at diagnosis and sex.

Tabular features (Task C)
-------------------------
Age, sex, race one-hot, 20 mutation flags plus derived biology-driven combinations.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from glioma.config import PROCESSED, RAW

CLINICAL = ["age", "male"]
EXCLUDED_ARMS = {"21p", "Xp", "Xq", "Yp", "Yq"}


@lru_cache(maxsize=1)
def arm_map() -> dict[str, list[str]]:
    g = pd.read_parquet(PROCESSED / "gene_annot.parquet").dropna(subset=["arm"])
    g = g[~g["arm"].isin(EXCLUDED_ARMS)]
    return {a: sorted(s) for a, s in g.groupby("arm")["symbol"]}


@lru_cache(maxsize=1)
def gene_sets() -> dict[str, list[str]]:
    sets = {}
    for path, keep in [(RAW / "msigdb/h.all.v2023.2.Hs.symbols.gmt", "HALLMARK_"),
                       (RAW / "msigdb/c2.cgp.v2023.2.Hs.symbols.gmt", "VERHAAK_GLIOBLASTOMA_")]:
        for line in path.read_text().splitlines():
            name, _, *genes = line.split("\t")
            if name.startswith(keep):
                sets[name] = genes
    return sets


class SampleRankNormalizer(BaseEstimator, TransformerMixin):
    """Within-sample rank -> [0, 1] for gene columns. Platform/composition invariant (stateless).

    Used only for cross-platform external validation (RNA-seq -> microarray).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        genes = [c for c in X.columns if c not in CLINICAL]
        out = X.copy()
        out[genes] = X[genes].rank(axis=1, pct=True).astype("float32")
        return out


class GliomaFeatures(BaseEstimator, TransformerMixin):
    """Expression (+clinical) -> engineered feature matrix. X must be a DataFrame."""

    def __init__(self, use_genes=True, k_genes=2000, use_arms=True, use_sets=True, use_clinical=True,
                 min_genes_per_arm=20, min_set_coverage=0.5):
        self.use_genes = use_genes
        self.k_genes = k_genes
        self.use_arms = use_arms
        self.use_sets = use_sets
        self.use_clinical = use_clinical
        self.min_genes_per_arm = min_genes_per_arm
        self.min_set_coverage = min_set_coverage

    def fit(self, X: pd.DataFrame, y=None):
        genes = [c for c in X.columns if c not in CLINICAL]
        G = X[genes].to_numpy(dtype="float32")
        self.genes_ = np.array(genes)
        self.mu_ = G.mean(0)
        sd = G.std(0)
        expressed = (self.mu_ > np.quantile(self.mu_, 0.20)) & (sd > 1e-6)  # drop lowest-expressed 20%
        self.sd_ = np.where(sd > 1e-6, sd, 1.0)
        var = np.where(expressed, sd ** 2, -1)
        k = min(self.k_genes, int(expressed.sum()))
        self.top_idx_ = np.sort(np.argsort(var)[::-1][:k])
        self.expressed_idx_ = np.flatnonzero(expressed)
        pos = {g: i for i, g in enumerate(genes)}
        ok = set(self.expressed_idx_)
        def usable(gs):  # indices of a gene list's genes that are present and expressed
            return [pos[g] for g in gs if g in pos and pos[g] in ok]

        self.arm_idx_ = {a: idx for a, gs in arm_map().items()
                         if len(idx := usable(gs)) >= self.min_genes_per_arm}
        self.set_idx_ = {s: idx for s, gs in gene_sets().items()
                         if len(idx := usable(gs)) >= self.min_set_coverage * len(gs)}
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = self._names()
        return self

    def _names(self) -> list[str]:
        names = []
        if self.use_genes:
            names += [f"gene:{g}" for g in self.genes_[self.top_idx_]]
        if self.use_arms:
            names += [f"arm:{a}" for a in self.arm_idx_]
        if self.use_sets:
            names += [f"set:{s.replace('HALLMARK_', 'H_').replace('VERHAAK_GLIOBLASTOMA_', 'VERHAAK_')}"
                      for s in self.set_idx_]
        if self.use_clinical:
            names += CLINICAL
        return names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        G = X[list(self.genes_)].to_numpy(dtype="float32")
        Z = (G - self.mu_) / self.sd_
        blocks = []
        if self.use_genes:
            blocks.append(Z[:, self.top_idx_])
        if self.use_arms:
            base = Z[:, self.expressed_idx_].mean(1, keepdims=True)
            blocks.append(np.column_stack([Z[:, i].mean(1) for i in self.arm_idx_.values()]) - base)
        if self.use_sets:
            blocks.append(np.column_stack([Z[:, i].mean(1) for i in self.set_idx_.values()]))
        if self.use_clinical:
            blocks.append(X[CLINICAL].to_numpy(dtype="float32"))
        return pd.DataFrame(np.hstack(blocks), index=X.index, columns=self.feature_names_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class TabularFeatures(BaseEstimator, TransformerMixin):
    """Task C: clinical + mutation flags, plus derived biologically-motivated combinations."""

    RACES = ["white", "black or african american", "asian"]

    def __init__(self, derived=True):
        self.derived = derived

    def fit(self, X, y=None):
        self.genes_ = [c for c in X.columns if c.isupper()]
        self.feature_names_ = list(self.transform(X.iloc[:2]).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[["age", "male"] + self.genes_].astype("float32").copy()
        for r in self.RACES:
            out[f"race:{r.split()[0]}"] = (X["race"] == r).astype("float32")
        if self.derived:
            m = X[self.genes_]
            out["idh_any"] = ((m["IDH1"] + m["IDH2"]) > 0).astype("float32")           # IDH-mutant lineage
            out["astro_sig"] = ((m["TP53"] + m["ATRX"]) == 2).astype("float32")      # TP53+ATRX: astrocytoma
            out["oligo_sig"] = ((m["CIC"] + m["FUBP1"]) > 0).astype("float32")      # CIC/FUBP1: oligodendroglioma
            out["gbm_sig"] = ((m["EGFR"] + m["PTEN"] + m["NF1"]) > 0).astype("float32")  # classic GBM drivers
            out["n_mut_genes"] = m.sum(1).astype("float32")                         # burden over panel
            out["age_over_50"] = (X["age"] > 50).astype("float32")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
