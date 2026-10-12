"""Turn raw downloads into tidy, analysis-ready parquet tables.

Outputs (data/processed/)
-------------------------
tcga_expr.parquet       samples x genes, log2(RSEM+1), float32, NCBI-current gene symbols
tcga_meta.parquet       per-patient clinical data + the three task labels (Ceccarelli 2016 Table S1)
tcga_mutations.parquet  binary protein-altering mutation flags (Tasci genes), exome-sequenced patients only
gene_annot.parquet      symbol -> Entrez ID, chromosome, arm (from NCBI Gene cytobands)
ext_<gse>_expr.parquet / ext_<gse>_meta.parquet   external GEO cohorts on the same gene namespace
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from glioma.config import PROCESSED, RAW
from glioma.download import TASCI_GENES

TASK_A, TASK_B, TASK_C = "subtype_3", "subtype_7", "grade_2"
TASKS = {
    TASK_A: "3-class WHO-2021 molecular subtype (IDH / 1p19q)",
    TASK_B: "7-class pan-glioma DNA-methylation subtype (Ceccarelli 2016)",
    TASK_C: "LGG vs GBM (Tasci 2022 clinical + mutation setup)",
}


# --------------------------------------------------------------------------- genes
def gene_annotation() -> pd.DataFrame:
    gi = pd.read_csv(RAW / "ncbi/Homo_sapiens.gene_info.gz", sep="\t",
                     usecols=["GeneID", "Symbol", "Synonyms", "chromosome", "map_location", "type_of_gene"])
    arm = gi["map_location"].str.extract(r"^(\d{1,2}|X|Y)([pq])")
    gi["chrom"] = arm[0]
    gi["arm"] = arm[0] + arm[1]
    return gi.rename(columns={"GeneID": "entrez", "Symbol": "symbol"})


def symbol_mapper(gi: pd.DataFrame) -> dict[str, int]:
    """Map current symbols, unique synonyms and bare Entrez IDs to an Entrez ID."""
    current = dict(zip(gi["symbol"], gi["entrez"]))
    syn = gi[["entrez", "Synonyms"]].assign(s=gi["Synonyms"].str.split("|")).explode("s")
    syn = syn[(syn["s"] != "-") & ~syn["s"].isin(current)]
    syn = syn.drop_duplicates(["s", "entrez"])
    syn = syn[~syn["s"].duplicated(keep=False)]            # ambiguous synonyms are dropped
    mapping = {**dict(zip(syn["s"], syn["entrez"])), **current}
    mapping.update({str(e): e for e in gi["entrez"]})
    return mapping


def collapse_to_symbols(df: pd.DataFrame, entrez: pd.Series, gi: pd.DataFrame) -> pd.DataFrame:
    """df: samples x features; entrez: feature -> Entrez ID. Returns samples x current symbol (mean)."""
    sym = dict(zip(gi["entrez"], gi["symbol"]))
    keep = entrez.dropna().astype(int)
    keep = keep[keep.isin(sym)]
    out = df[keep.index].T.groupby(keep.map(sym).values).mean().T
    return out.astype("float32")


# --------------------------------------------------------------------------- TCGA
def build_tcga(gi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW / "gdc_pub/LGG-GBM.gene_expression.normalized.txt", sep="\t", index_col=0)
    raw.index = raw.index.str[:12]
    raw = raw[~raw.index.duplicated()]
    mapper = symbol_mapper(gi)
    entrez = pd.Series({c: mapper.get(str(c)) for c in raw.columns})
    print(f"TCGA expression: {raw.shape[0]} samples, {raw.shape[1]} genes; "
          f"{entrez.notna().mean():.1%} of symbols mapped to NCBI Gene")
    expr = collapse_to_symbols(np.log2(raw.clip(lower=0) + 1), entrez, gi)

    s1 = pd.read_excel(RAW / "gdc_pub/TableS1.PatientData.20151020.v3.xlsx", sheet_name=0, header=1)
    cols = {
        "Case": "patient", "Tissue source site": "tss", "Study": "study", "Whole exome": "has_wes",
        "RNAseq": "has_rnaseq", "Histology": "histology", "Grade": "who_grade",
        "Age (years at diagnosis)": "age", "Gender": "gender", "Survival (months)": "os_months",
        "Vital status (1=dead)": "os_event", "Karnofsky Performance Score": "kps",
        "Mutation Count": "mutation_count", "Percent aneuploidy": "pct_aneuploidy",
        "IDH status": "idh_status", "1p/19q codeletion": "codel_1p19q", "IDH/codel subtype": TASK_A,
        "MGMT promoter status": "mgmt", "Chr 7 gain/Chr 10 loss": "chr7gain_chr10loss",
        "TERT promoter status": "tert_promoter", "ATRX status": "atrx_status",
        "Transcriptome Subtype": "verhaak_subtype", "Supervised DNA Methylation Cluster": TASK_B,
        "ESTIMATE stromal score": "estimate_stromal", "ESTIMATE immune score": "estimate_immune",
        "ABSOLUTE purity": "purity",
    }
    meta = s1[list(cols)].rename(columns=cols).set_index("patient")
    meta = meta.replace({"NA": np.nan, "[Not Available]": np.nan})
    for c in ["age", "os_months", "kps", "mutation_count", "pct_aneuploidy", "estimate_stromal",
              "estimate_immune", "purity"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta[TASK_C] = meta["study"].map({"Brain Lower Grade Glioma": "LGG", "Glioblastoma multiforme": "GBM"})
    meta["batch_tss"] = meta.index.str[5:7]

    demo = pd.read_parquet(RAW / "gdc_api/tcga_glioma_demographics.parquet").set_index("patient")
    meta["race"] = demo["race"].reindex(meta.index).replace({"not reported": np.nan, "Unknown": np.nan})
    meta["in_expr"] = meta.index.isin(expr.index)

    # mutations: exome-sequenced patients get 0 for genes with no protein-altering call
    mut = pd.read_parquet(RAW / "gdc_api/tcga_glioma_mutations.parquet")
    sequenced = sorted(set(meta.index[meta["has_wes"] == "Yes"]) | set(mut["patient"]))
    mm = (pd.crosstab(mut["patient"], mut["gene"]).clip(upper=1)
          .reindex(index=sequenced, columns=TASCI_GENES, fill_value=0).astype("int8"))
    expr = expr.loc[expr.index.isin(meta.index)]
    return expr, meta, mm


# --------------------------------------------------------------------------- GEO
def _read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chars: list[list[str]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                gsm = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
        expr = pd.read_csv(f, sep="\t", index_col=0, comment="!")
    meta = pd.DataFrame(index=gsm)
    for row in chars:
        for g, v in zip(gsm, row):
            if ": " in v:
                k, val = v.split(": ", 1)
                meta.loc[g, k.strip()] = val.strip()
    expr.index = expr.index.astype(str)
    return expr.T, meta


def _platform_table(path: Path, cols: list[str]) -> pd.DataFrame:
    """Stream the probe table (between !platform_table_begin/end) out of a GEO platform text/SOFT file."""
    rows, header = [], None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                header = next(f).rstrip("\n").split("\t")
                idx = [header.index(c) for c in cols]
                continue
            if header is not None:
                if line.startswith("!platform_table_end"):
                    break
                parts = line.rstrip("\n").split("\t")
                rows.append([parts[i] if i < len(parts) else "" for i in idx])
    return pd.DataFrame(rows, columns=cols)


def build_gse107850(gi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr, meta = _read_series_matrix(RAW / "geo/GSE107850_series_matrix.txt.gz")
    plat = _platform_table(RAW / "geo/GPL14951_table.txt", ["ID", "Entrez_Gene_ID"]).set_index("ID")
    entrez = pd.to_numeric(plat["Entrez_Gene_ID"], errors="coerce").reindex(expr.columns)
    expr = collapse_to_symbols(expr, entrez, gi)
    meta = meta.rename(columns={"idh.status": "idh_status", "age": "age", "gender": "gender"})
    meta["idh_status"] = meta["idh_status"].map({"mutated": "Mutant", "normal": "WT"})
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["histology_raw"] = meta["histology"]
    meta["histology"] = meta["histology"].str.extract(r"^(\w+)")[0].map(
        {"AOD": "oligodendroglioma", "AOA": "oligoastrocytoma", "AA": "astrocytoma"})
    return expr, meta


def build_gse16011(gi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr, meta = _read_series_matrix(RAW / "geo/GSE16011_series_matrix.txt.gz")
    entrez = pd.to_numeric(pd.Series(expr.columns, index=expr.columns).str.replace("_at", "", regex=False),
                           errors="coerce")
    expr = collapse_to_symbols(expr, entrez, gi)
    meta = meta[meta["histology"] != "control"].copy()
    meta["age"] = pd.to_numeric(meta.get("age at diagnosis"), errors="coerce")
    h = meta["histology"].str.extract(r"^(\w+)\s*\(grade (\w+)\)")
    meta["histology_raw"] = meta["histology"]
    meta["histology"] = h[0].map({"OD": "oligodendroglioma", "GBM": "glioblastoma", "A": "astrocytoma",
                                  "OA": "oligoastrocytoma", "PA": "pilocytic astrocytoma"})
    meta["who_grade"] = h[1]
    return expr.loc[meta.index], meta


# --------------------------------------------------------------------------- main
def main() -> None:
    gi = gene_annotation()
    gi[["symbol", "entrez", "chrom", "arm", "map_location", "type_of_gene"]].to_parquet(
        PROCESSED / "gene_annot.parquet")

    expr, meta, mut = build_tcga(gi)
    expr.to_parquet(PROCESSED / "tcga_expr.parquet")
    meta.to_parquet(PROCESSED / "tcga_meta.parquet")
    mut.to_parquet(PROCESSED / "tcga_mutations.parquet")
    print(f"tcga_expr {expr.shape}, tcga_meta {meta.shape}, tcga_mutations {mut.shape}")
    with_expr = meta.loc[meta["in_expr"]]
    for t, desc in TASKS.items():
        base = with_expr if t != TASK_C else meta.loc[meta.index.isin(mut.index)]
        print(f"\n{t}: {desc}\n{base[t].value_counts().to_string()}")

    for name, fn in [("gse107850", build_gse107850), ("gse16011", build_gse16011)]:
        e, m = fn(gi)
        e.to_parquet(PROCESSED / f"ext_{name}_expr.parquet")
        m.to_parquet(PROCESSED / f"ext_{name}_meta.parquet")
        shared = len(set(e.columns) & set(expr.columns))
        print(f"\n{name}: {e.shape[0]} samples, {e.shape[1]} genes ({shared} shared with TCGA)")
        print(m["histology"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
