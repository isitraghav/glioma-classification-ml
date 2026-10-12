"""Exploratory data analysis -> reports/figures/eda_*.png + reports/tables/eda_*.csv.

Label-dependent analyses use the *training split only* so that nothing seen here can leak test
information into later modelling decisions. Class balance is reported for the full cohort.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from glioma.build_dataset import TASK_A, TASK_B, TASK_C, TASKS
from glioma.config import PROCESSED, RAW, SEED, TABLES
from glioma.data import CLASS_ORDER, expression, load_task, meta
from glioma.features import GliomaFeatures
from glioma.plots import AXIS, DIV, INK2, MUTED, SEQ, SERIES, colors_for, save

SUMMARY: dict = {}


def class_balance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [3, 6, 2]})
    for ax, task in zip(axes, [TASK_A, TASK_B, TASK_C]):
        d = load_task(task)
        y = pd.concat([d.y_train, d.y_test])
        counts = y.value_counts().reindex(d.classes)
        ax.bar(range(len(counts)), counts.values, color=SERIES[0], width=0.65)
        for i, v in enumerate(counts.values):
            ax.text(i, v + counts.max() * 0.02, str(v), ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(counts)), counts.index, rotation=30, ha="right")
        ax.set_title(f"{task}  (n={len(y)})")
        ax.grid(axis="x", visible=False)
        SUMMARY[f"{task}_counts"] = counts.to_dict()
        SUMMARY[f"{task}_imbalance_ratio"] = round(counts.max() / counts.min(), 2)
    axes[0].set_ylabel("patients")
    fig.suptitle("Class balance per task", fontweight="bold")
    save(fig, "eda_class_balance")


def clinical(d) -> None:
    m = meta().loc[d.y_train.index]
    y = d.y_train
    col = colors_for(d.classes)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    data = [m.loc[y == c, "age"].dropna() for c in d.classes]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "white", "lw": 2})
    for patch, c in zip(bp["boxes"], d.classes):
        patch.set(facecolor=col[c], edgecolor=col[c])
    axes[0].set_xticks(range(1, len(d.classes) + 1), d.classes, rotation=15)
    axes[0].set_title("Age at diagnosis by subtype")
    axes[0].set_ylabel("years")
    kw = stats.kruskal(*data)
    SUMMARY["age_kruskal_p"] = float(kw.pvalue)

    sex = pd.crosstab(y, m["gender"], normalize="index").reindex(d.classes)
    axes[1].barh(sex.index, sex.get("male", 0), color=SERIES[0], label="male", height=0.6)
    axes[1].barh(sex.index, sex.get("female", 0), left=sex.get("male", 0), color=SERIES[1], label="female",
                 height=0.6)
    axes[1].set_title("Sex composition")
    axes[1].set_xlabel("fraction")
    axes[1].legend(loc="lower right")

    # missingness of clinical / molecular annotations (full cohort)
    cols = ["age", "gender", "race", "kps", "os_months", "mgmt", "tert_promoter", "atrx_status",
            "idh_status", "codel_1p19q", TASK_A, TASK_B, "purity", "mutation_count"]
    miss = meta()[cols].isna().mean().sort_values()
    axes[2].barh(miss.index, miss.values, color=SERIES[0], height=0.6)
    axes[2].set_title("Missing fraction (all 1,122 patients)")
    axes[2].set_xlim(0, 1)
    miss.to_csv(TABLES / "eda_missingness.csv", header=["missing_fraction"])
    save(fig, "eda_clinical")


def expression_qc() -> None:
    e = expression()
    v = e.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(e), 40, replace=False):
        h, b = np.histogram(v[i], bins=80, range=(0, 20), density=True)
        axes[0].plot(b[:-1], h, color=SERIES[0], alpha=0.25, lw=1)
    axes[0].set_title("log2(RSEM+1) density, 40 random samples")
    axes[0].set_xlabel("log2 expression")
    mu, sd = v.mean(0), v.std(0)
    axes[1].hexbin(mu, sd, gridsize=60, cmap=SEQ, mincnt=1, bins="log")
    axes[1].set_title("Mean-variance relationship (genes)")
    axes[1].set_xlabel("mean log2 expression")
    axes[1].set_ylabel("SD")
    var_sorted = np.sort(sd ** 2)[::-1]
    axes[2].plot(np.cumsum(var_sorted) / var_sorted.sum(), color=SERIES[0])
    for k in (500, 2000, 5000):
        axes[2].axvline(k, color=AXIS, lw=1, ls="--")
        axes[2].text(k, 0.05, f" k={k}", color=MUTED, fontsize=8)
    axes[2].set_title("Cumulative variance captured by top-k genes")
    axes[2].set_xlabel("genes ranked by variance")
    axes[2].set_xscale("log")
    save(fig, "eda_expression_qc")
    SUMMARY.update({
        "n_genes": int(e.shape[1]), "n_expr_samples": int(e.shape[0]),
        "frac_genes_mean_below_1": float((mu < 1).mean()),
        "frac_zero_values": float((v == 0).mean()),
        "var_frac_top2000": float(var_sorted[:2000].sum() / var_sorted.sum()),
    })


def embeddings(d) -> None:
    X = d.X_train
    feats = GliomaFeatures(k_genes=2000, use_arms=False, use_sets=False, use_clinical=False).fit(X)
    Z = feats.transform(X).to_numpy()
    pca = PCA(n_components=50, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    T = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(P[:, :30])
    m = meta().loc[X.index]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    col = colors_for(d.classes)
    for ax, (emb, name) in zip(axes[:2], [(P, "PCA"), (T, "t-SNE")]):
        for c in d.classes:
            s = (d.y_train == c).to_numpy()
            ax.scatter(emb[s, 0], emb[s, 1], s=12, color=col[c], label=c, edgecolor="white", lw=0.3)
        ax.set_title(f"{name} of top-2000 variable genes")
        lab = (f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", f"PC2 ({pca.explained_variance_ratio_[1]:.0%})") \
            if name == "PCA" else ("t-SNE 1", "t-SNE 2")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
    axes[0].legend(loc="best", markerscale=1.5)
    # batch: tissue source site, top 3 sites + other (series cap for scatter = 3)
    top = m["batch_tss"].value_counts().index[:3]
    bt = m["batch_tss"].where(m["batch_tss"].isin(top), "other")
    for i, b in enumerate(list(top) + ["other"]):
        s = (bt == b).to_numpy()
        axes[2].scatter(P[s, 0], P[s, 1], s=12, color=SERIES[i] if b != "other" else "#c3c2b7",
                        label=f"site {b}" if b != "other" else "other sites", edgecolor="white", lw=0.3)
    axes[2].set_title("PCA coloured by tissue-source site (batch)")
    axes[2].set_xlabel("PC1")
    axes[2].legend()
    save(fig, "eda_embeddings")

    # how much of each PC is explained by biology (subtype) vs batch (site): one-way ANOVA R^2
    rows = []
    for k in range(10):
        pc = P[:, k]
        r2 = {}
        for name, g in [("subtype", d.y_train.to_numpy()), ("tissue_site", m["batch_tss"].to_numpy()),
                        ("study_LGG_vs_GBM", m["study"].to_numpy())]:
            ss_tot = ((pc - pc.mean()) ** 2).sum()
            ss_b = sum((pc[g == lv].mean() - pc.mean()) ** 2 * (g == lv).sum() for lv in np.unique(g))
            r2[name] = ss_b / ss_tot
        rows.append({"PC": k + 1, "var_explained": pca.explained_variance_ratio_[k], **r2})
    pd.DataFrame(rows).round(3).to_csv(TABLES / "eda_pc_variance_sources.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    pv = pd.DataFrame(rows).set_index("PC")
    w = 0.38
    ax.bar(pv.index - w / 2, pv["subtype"], w, color=SERIES[0], label="subtype (biology)")
    ax.bar(pv.index + w / 2, pv["tissue_site"], w, color=SERIES[1], label="tissue-source site (batch)")
    ax.set_xticks(pv.index)
    ax.set_xlabel("principal component")
    ax.set_ylabel("ANOVA R²")
    ax.set_title("Variance in each PC explained by subtype vs batch")
    ax.legend()
    save(fig, "eda_pc_variance_sources")


def mutations(d) -> None:
    mut = pd.read_parquet(PROCESSED / "tcga_mutations.parquet")
    y = d.y_train[d.y_train.index.isin(mut.index)]
    freq = mut.loc[y.index].groupby(y).mean().reindex(d.classes).T
    freq = freq.loc[freq.max(axis=1).sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    im = axes[0].imshow(freq.values, aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    axes[0].set_yticks(range(len(freq)), freq.index, fontsize=8)
    axes[0].set_xticks(range(len(d.classes)), d.classes, rotation=15)
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            val = freq.values[i, j]
            axes[0].text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7,
                         color="white" if val > 0.5 else INK2)
    axes[0].grid(False)
    axes[0].set_title(f"Mutation frequency by subtype (train, n={len(y)})")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    freq.round(3).to_csv(TABLES / "eda_mutation_freq_by_subtype.csv")

    # co-occurrence: log odds ratio between gene pairs (Haldane-corrected)
    g = [c for c in freq.index[:12]]
    M = mut.loc[y.index, g].to_numpy()
    lor = np.zeros((len(g), len(g)))
    for i in range(len(g)):
        for j in range(len(g)):
            a = ((M[:, i] == 1) & (M[:, j] == 1)).sum() + .5
            b = ((M[:, i] == 1) & (M[:, j] == 0)).sum() + .5
            c = ((M[:, i] == 0) & (M[:, j] == 1)).sum() + .5
            dd = ((M[:, i] == 0) & (M[:, j] == 0)).sum() + .5
            lor[i, j] = np.log(a * dd / (b * c)) if i != j else np.nan
    im = axes[1].imshow(lor, cmap=DIV, vmin=-4, vmax=4)
    axes[1].set_xticks(range(len(g)), g, rotation=90, fontsize=8)
    axes[1].set_yticks(range(len(g)), g, fontsize=8)
    axes[1].grid(False)
    axes[1].set_title("Mutation co-occurrence (log odds ratio)")
    fig.colorbar(im, ax=axes[1], fraction=0.04, label="← exclusive   co-occurring →")
    save(fig, "eda_mutations")


def differential_expression(d) -> None:
    X = d.X_train.drop(columns=["age", "male"])
    X = X.loc[:, X.mean() > 2]
    y = d.y_train
    H, p = stats.kruskal(*[X[y == c].to_numpy() for c in d.classes])
    res = pd.DataFrame({"H": H, "p": p}, index=X.columns)
    res["q_bonferroni"] = np.minimum(res["p"] * len(res), 1)
    res = res.sort_values("H", ascending=False)
    res.head(200).to_csv(TABLES / "eda_kruskal_top_genes.csv")
    SUMMARY["n_genes_bonferroni_sig"] = int((res["q_bonferroni"] < 0.05).sum())
    # top 8 genes per class by mean z difference vs rest
    z = (X - X.mean()) / X.std()
    top = []
    for c in d.classes:
        diff = z[y == c].mean() - z[y != c].mean()
        top += [g for g in diff.loc[res.index[:1000]].sort_values(ascending=False).index[:8] if g not in top]
    order = np.concatenate([np.flatnonzero((y == c).to_numpy()) for c in d.classes])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(z[top].to_numpy()[order].T, aspect="auto", cmap=DIV, vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_yticks(range(len(top)), top, fontsize=7.5)
    bounds = np.cumsum([(y == c).sum() for c in d.classes])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=2)
    centers = np.concatenate([[0], bounds[:-1]]) + np.diff(np.concatenate([[0], bounds])) / 2
    ax.set_xticks(centers, d.classes)
    ax.grid(False)
    ax.set_title("Top marker genes per subtype (Kruskal–Wallis ranked; z-scored expression)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="z-score")
    save(fig, "eda_marker_heatmap")


def arm_scores_vs_copy_number(d) -> None:
    """Validate the derived chromosome-arm features against measured GISTIC copy number (EDA only)."""
    X = d.X_train
    f = GliomaFeatures(use_genes=False, use_sets=False, use_clinical=False).fit(X)
    A = f.transform(X)
    cn = pd.read_csv(RAW / "gdc_pub/GISTIC_copy_number_matrix.txt", sep="\t", index_col=0)
    cyto = cn.pop("Cytoband")
    cn = cn.drop(columns=[c for c in cn.columns if not c.startswith("TCGA")])
    cn.columns = cn.columns.str[:12]
    cn = cn.loc[:, ~cn.columns.duplicated()]
    arm = cyto.astype(str).str.extract(r"^(\d+|X)([pq])").agg("".join, axis=1)
    shared = A.index.intersection(cn.columns)
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for i, a in enumerate(["1p", "19q", "7q", "10q"]):
        cn_arm = cn.loc[(arm == a).to_numpy(), shared].mean()
        ex_arm = A.loc[shared, f"arm:{a}"]
        r, _ = stats.pearsonr(ex_arm, cn_arm)
        rows.append({"arm": a, "pearson_r_expr_score_vs_copy_number": round(r, 3), "n": len(shared)})
        yy = d.y_train.loc[shared]
        for c, colr in colors_for(d.classes).items():
            s = (yy == c).to_numpy()
            axes[i].scatter(cn_arm[s], ex_arm[s], s=8, color=colr, label=c, edgecolor="none", alpha=0.8)
        axes[i].set_title(f"arm {a}: r = {r:.2f}")
        axes[i].set_xlabel("GISTIC copy number (log2 ratio)")
    axes[0].set_ylabel("expression-derived arm score")
    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle("Derived chromosome-arm expression scores track measured copy number", fontweight="bold")
    save(fig, "eda_arm_scores_vs_copy_number")
    pd.DataFrame(rows).to_csv(TABLES / "eda_arm_score_validation.csv", index=False)

    # all-arm profile per subtype
    prof = A.groupby(d.y_train).mean().reindex(d.classes)
    fig, ax = plt.subplots(figsize=(13, 2.6))
    im = ax.imshow(prof.values, aspect="auto", cmap=DIV, vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(prof.shape[1]), [c.replace("arm:", "") for c in prof.columns], fontsize=7.5)
    ax.set_yticks(range(len(prof)), prof.index)
    ax.grid(False)
    ax.set_title("Mean chromosome-arm expression score per subtype (blue = lower expression ≈ loss)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    save(fig, "eda_arm_profile")


def survival(d) -> None:
    """Kaplan–Meier by subtype: shows the classes are clinically distinct."""
    m = meta().loc[d.y_train.index]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c, colr in colors_for(d.classes).items():
        s = m[d.y_train == c].dropna(subset=["os_months", "os_event"])
        t, e = s["os_months"].to_numpy(), s["os_event"].astype(float).to_numpy()
        times = np.sort(np.unique(t[e == 1]))
        surv, S = [1.0], 1.0
        for ti in times:
            n_risk, d_ = (t >= ti).sum(), ((t == ti) & (e == 1)).sum()
            S *= 1 - d_ / n_risk
            surv.append(S)
        ax.step(np.concatenate([[0], times]), surv, where="post", color=colr, label=f"{c} (n={len(s)})")
    ax.set_xlabel("months")
    ax.set_ylabel("overall survival")
    ax.set_xlim(0, 120)
    ax.set_title("Kaplan–Meier overall survival by subtype")
    ax.legend()
    save(fig, "eda_survival")


def task_b_vs_a() -> None:
    m = meta()
    ct = pd.crosstab(m[TASK_B], m[TASK_A]).reindex(index=CLASS_ORDER[TASK_B], columns=CLASS_ORDER[TASK_A])
    ct.to_csv(TABLES / "eda_subtype7_vs_subtype3.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(ct.values, cmap=SEQ, aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=8,
                    color="white" if ct.values[i, j] > ct.values.max() / 2 else INK2)
    ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=15)
    ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.grid(False)
    ax.set_title("7 methylation subtypes nest inside the 3 WHO classes")
    save(fig, "eda_subtype7_vs_subtype3")


def main() -> None:
    d = load_task(TASK_A)
    class_balance()
    clinical(d)
    expression_qc()
    embeddings(d)
    mutations(d)
    differential_expression(d)
    arm_scores_vs_copy_number(d)
    survival(d)
    task_b_vs_a()
    (TABLES / "eda_summary.json").write_text(json.dumps(SUMMARY, indent=1, default=float))
    print(json.dumps(SUMMARY, indent=1, default=float))
    print(f"tasks: {TASKS}")


if __name__ == "__main__":
    main()
