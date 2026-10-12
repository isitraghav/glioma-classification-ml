"""Download raw data from official sources (aria2c, concurrent) and record provenance.

Sources
-------
* NCI Genomic Data Commons (GDC) publication page for Ceccarelli et al. 2016,
  Cell 164:550-563 (TCGA pan-glioma): https://gdc.cancer.gov/about-data/publications/lgggbm_2016
  - RNASeqV2 expression matrix, Table S1 (patient labels/clinical), GISTIC copy number.
* GDC REST API: open-access somatic mutation calls (ssm_occurrences) for TCGA-LGG/GBM.
* NCBI GEO (external cohorts): GSE107850 (EORTC 22033 LGG trial, IDH status),
  GSE16011 (Gravendeel 2009, 284 gliomas with histology) + the GPL14951 probe table.
* NCBI Gene: Homo_sapiens.gene_info (symbol synonyms + cytoband -> chromosome arm).
* MSigDB (Broad Institute): Hallmark and curated (C2/CGP, incl. Verhaak 2010 GBM subtypes) gene sets.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests

from glioma.config import GDC_API, GDC_DATA, RAW

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"
MSIGDB = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/"

# relative destination -> (url, description)
FILES: dict[str, tuple[str, str]] = {
    "gdc_pub/LGG-GBM.gene_expression.normalized.txt": (
        GDC_DATA + "0759ea74-25fa-4249-aabc-ae10617dda28",
        "TCGA LGG+GBM RNASeqV2 RSEM-normalised expression matrix (Ceccarelli 2016 freeze)"),
    "gdc_pub/TableS1.PatientData.20151020.v3.xlsx": (
        GDC_DATA + "a0886693-4c6e-40de-9cb1-8bfcb4bb30a7",
        "Ceccarelli 2016 Table S1: per-patient clinical + molecular labels (IDH, 1p/19q, subtypes)"),
    "gdc_pub/GISTIC_copy_number_matrix.txt": (
        GDC_DATA + "9ed4e8f3-9354-455b-90da-4532656c36db",
        "Ceccarelli 2016 GISTIC copy-number matrix (EDA only, never a Task-A feature)"),
    "geo/GSE107850_series_matrix.txt.gz": (
        f"{GEO_FTP}/series/GSE107nnn/GSE107850/matrix/GSE107850_series_matrix.txt.gz",
        "GEO GSE107850: EORTC 22033-26033 LGG trial expression (Illumina WG-DASL), IDH status"),
    # platform table only (~62 MB) instead of the 2.4 GB "family" SOFT file that bundles every sample.
    # GSE16011's platform (GPL8542) needs no annotation: its probe IDs are "<Entrez ID>_at".
    "geo/GPL14951_table.txt": (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL14951&targ=self&form=text&view=data",
        "GEO platform GPL14951 probe table (Illumina HumanHT-12 WG-DASL V4)"),
    "geo/GSE16011_series_matrix.txt.gz": (
        f"{GEO_FTP}/series/GSE16nnn/GSE16011/matrix/GSE16011_series_matrix.txt.gz",
        "GEO GSE16011: Gravendeel 2009, 284 gliomas (Affymetrix U133 Plus 2, Entrez CDF), histology"),
    "ncbi/Homo_sapiens.gene_info.gz": (
        "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
        "NCBI Gene: human gene symbols, synonyms, Entrez IDs, cytobands"),
    "msigdb/h.all.v2023.2.Hs.symbols.gmt": (
        MSIGDB + "h.all.v2023.2.Hs.symbols.gmt", "MSigDB Hallmark gene sets"),
    "msigdb/c2.cgp.v2023.2.Hs.symbols.gmt": (
        MSIGDB + "c2.cgp.v2023.2.Hs.symbols.gmt", "MSigDB C2 CGP gene sets (Verhaak 2010 GBM subtypes)"),
}

# Genes used by Tasci et al. 2022 (UCI #759) for LGG-vs-GBM grading.
TASCI_GENES = ["IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1",
               "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4", "PDGFRA"]
PROTEIN_ALTERING = {
    "missense_variant", "stop_gained", "frameshift_variant", "inframe_deletion", "inframe_insertion",
    "splice_acceptor_variant", "splice_donor_variant", "start_lost", "stop_lost", "protein_altering_variant",
}

MANIFEST = RAW / "MANIFEST.csv"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(rows: list[tuple[Path, str, str]]) -> None:
    done = set()
    if MANIFEST.exists():
        done = set(pd.read_csv(MANIFEST)["file"])
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "url", "description", "downloaded_utc", "bytes", "md5"])
        for path, url, desc in rows:
            rel = path.relative_to(RAW).as_posix()
            if rel in done or not path.exists():
                continue
            w.writerow([rel, url, desc, dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                        path.stat().st_size, _md5(path)])


USER_AGENT = "glioma-ml-course-project/0.1 (academic research)"  # NCBI blocks aria2's default agent


def _is_error_page(path: Path) -> bool:
    """Servers sometimes answer with an HTML error/blocked page instead of the data file."""
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html"))


def aria2_fetch(files: dict[str, tuple[str, str]], jobs: int = 8, conns: int = 8) -> None:
    """Download every missing file concurrently with aria2c (resumable, verified, then recorded)."""
    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c not found: install it (e.g. `sudo apt install aria2`).")
    todo = {rel: v for rel, v in files.items() if not (RAW / rel).exists()}
    if todo:
        lines = []
        for rel, (url, _) in todo.items():
            dest = RAW / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            lines += [url, f"  dir={dest.parent}", f"  out={dest.name}"]
            if "acc.cgi" in url:  # dynamic NCBI page: one polite connection, no range splitting
                lines += ["  split=1", "  max-connection-per-server=1"]
        listing = RAW / ".aria2_input.txt"
        listing.write_text("\n".join(lines) + "\n")
        cmd = ["aria2c", f"--input-file={listing}", f"--max-concurrent-downloads={jobs}",
               f"--split={conns}", f"--max-connection-per-server={conns}", "--min-split-size=5M",
               f"--user-agent={USER_AGENT}", "--continue=true", "--auto-file-renaming=false",
               "--allow-overwrite=true", "--max-tries=5", "--retry-wait=5", "--console-log-level=warn",
               "--summary-interval=0"]
        subprocess.run(cmd, check=True)
        listing.unlink()
    bad = [rel for rel in files if (RAW / rel).exists() and not rel.endswith((".gz", ".xlsx"))
           and _is_error_page(RAW / rel)]
    for rel in bad:
        (RAW / rel).unlink()
    if bad:
        raise SystemExit(f"server returned an HTML error page instead of data for: {bad} — re-run later")
    _record([(RAW / rel, url, desc) for rel, (url, desc) in files.items()])


def fetch_gdc_mutations(dest: Path = RAW / "gdc_api" / "tcga_glioma_mutations.parquet") -> Path:
    """Open-access somatic mutations for TCGA-LGG/GBM restricted to the Tasci genes.

    One row per (patient, gene) with a protein-altering mutation on the canonical transcript.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "case.project.project_id", "value": ["TCGA-LGG", "TCGA-GBM"]}},
        {"op": "in", "content": {"field": "ssm.consequence.transcript.gene.symbol", "value": TASCI_GENES}},
    ]}
    fields = ("case.submitter_id,ssm.ssm_id,ssm.consequence.transcript.gene.symbol,"
              "ssm.consequence.transcript.consequence_type,ssm.consequence.transcript.is_canonical")
    rows, start, size = [], 0, 2000
    while True:
        r = requests.post(GDC_API + "ssm_occurrences",
                          json={"filters": filt, "fields": fields, "size": size, "from": start}, timeout=180)
        r.raise_for_status()
        data = r.json()["data"]
        for hit in data["hits"]:
            pid = hit["case"]["submitter_id"]
            for c in hit["ssm"]["consequence"]:
                t = c["transcript"]
                if t.get("is_canonical") and t["gene"]["symbol"] in TASCI_GENES \
                        and t.get("consequence_type") in PROTEIN_ALTERING:
                    rows.append((pid, t["gene"]["symbol"], t["consequence_type"], hit["ssm"]["ssm_id"]))
        start += size
        if start >= data["pagination"]["total"]:
            break
    df = pd.DataFrame(rows, columns=["patient", "gene", "consequence", "ssm_id"]).drop_duplicates()
    df.to_parquet(dest)
    _record([(dest, GDC_API + "ssm_occurrences  filters=" + json.dumps(filt),
              "GDC open-access somatic mutations (canonical, protein-altering) for Tasci genes")])
    return dest


def fetch_gdc_demographics(dest: Path = RAW / "gdc_api" / "tcga_glioma_demographics.parquet") -> Path:
    """Race / ethnicity / sex / primary diagnosis for every TCGA-LGG/GBM case (GDC /cases)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    filt = {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-LGG", "TCGA-GBM"]}}
    fields = ("submitter_id,project.project_id,demographic.race,demographic.ethnicity,demographic.gender,"
              "diagnoses.age_at_diagnosis,diagnoses.primary_diagnosis")
    r = requests.post(GDC_API + "cases", json={"filters": filt, "fields": fields, "size": 5000}, timeout=180)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        demo, diag = h.get("demographic", {}), (h.get("diagnoses") or [{}])[0]
        rows.append({"patient": h["submitter_id"], "project": h["project"]["project_id"],
                     "race": demo.get("race"), "ethnicity": demo.get("ethnicity"), "gender": demo.get("gender"),
                     "age_at_diagnosis_days": diag.get("age_at_diagnosis"),
                     "primary_diagnosis": diag.get("primary_diagnosis")})
    pd.DataFrame(rows).to_parquet(dest)
    _record([(dest, GDC_API + "cases  filters=" + json.dumps(filt), "GDC case demographics for TCGA-LGG/GBM")])
    return dest


def main() -> None:
    aria2_fetch(FILES)
    fetch_gdc_mutations()
    fetch_gdc_demographics()
    print(pd.read_csv(MANIFEST)[["file", "bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
