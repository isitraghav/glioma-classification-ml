"""Scrape the literature (NCBI PubMed E-utilities) for related glioma-classification work.

Produces
  reports/tables/literature.csv          one row per paper (PMID, year, journal, DOI, title, abstract)
  reports/tables/literature_metrics.csv  every accuracy / AUC / F1 figure quoted in an abstract, with
                                         the sentence it came from, for comparison with our results.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from glioma.config import RAW, TABLES

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = {"tool": "glioma-ml-course-project"}

QUERIES = {
    "idh_codel_expression": '(glioma) AND (IDH) AND (1p/19q OR codeletion) AND (machine learning OR classifier) '
                            'AND (gene expression OR transcriptome OR RNA-seq)',
    "tcga_glioma_ml": '(TCGA) AND (glioma OR glioblastoma OR "lower grade glioma") AND '
                      '(machine learning OR random forest OR support vector OR XGBoost) AND classification',
    "glioma_grading_mutation": '(glioma grading) AND (mutation OR clinical features) AND (machine learning)',
    "pan_glioma_subtypes": '(pan-glioma OR "G-CIMP") AND (subtype) AND (classifier OR classification)',
    "brain_tumor_expression_ml": '(brain tumor OR brain tumour) AND (gene expression) AND '
                                 '(machine learning OR deep learning) AND classification',
    # the paper behind the UCI "Glioma Grading Clinical and Mutation Features" dataset (our Task C baseline)
    "tasci_2022": '(Tasci E[Author]) AND (glioma grading) AND (feature selection)',
}
KEY_PMIDS = {"26824661": "Ceccarelli 2016 Cell (data source)",
             "20129251": "Verhaak 2010 Cancer Cell (GBM expression subtypes)"}

# Directly comparable baselines: (pmid, study, metric, regex capturing the TCGA figure in the abstract).
# Values are always read from the scraped abstract, never typed in by hand.
BASELINES = [
    ("36430631", "Tasci et al. 2022, Int J Mol Sci — hierarchical voting feature selection + soft-voting ensemble "
                 "(source of UCI #759)", "accuracy (TCGA LGG vs GBM)",
     r"achieves ([\d.]+)% and [\d.]+% accuracy rates on TCGA"),
    ("37760597", "Tasci et al. 2023 (GradWise), Cancers — mRMR + LASSO hybrid feature selection",
     "accuracy (TCGA LGG vs GBM)", r"accuracy rate of ([\d.]+)% with \d+ features and an accuracy rate of [\d.]+% "
                                   r"with \w+ features on the TCGA"),
]

METRIC_RE = re.compile(
    r"(?P<metric>accuracy|AUC|area under the (?:ROC )?curve|F1(?:-score)?|sensitivity|specificity|"
    r"balanced accuracy|precision|recall)[^.;]{0,60}?(?P<value>\d{1,3}(?:\.\d+)?)\s?(?P<pct>%)?",
    re.IGNORECASE)


def esearch(term: str, retmax: int = 60) -> list[str]:
    r = requests.get(EUTILS + "esearch.fcgi", params={"db": "pubmed", "term": term, "retmax": retmax,
                                                     "sort": "relevance", "retmode": "json", **TOOL}, timeout=60)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def efetch(pmids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(pmids), 100):
        r = requests.get(EUTILS + "efetch.fcgi", params={"db": "pubmed", "id": ",".join(pmids[i:i + 100]),
                                                        "retmode": "xml", **TOOL}, timeout=120)
        r.raise_for_status()
        (RAW / "literature").mkdir(parents=True, exist_ok=True)
        (RAW / "literature" / f"pubmed_batch_{i // 100}.xml").write_bytes(r.content)
        for art in ET.fromstring(r.content).iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            abstract = " ".join("".join(a.itertext()) for a in art.iter("AbstractText"))
            doi = next((e.text for e in art.iter("ArticleId") if e.get("IdType") == "doi"), None)
            year = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//ArticleDate/Year")
            out.append({"pmid": pmid, "year": year, "journal": art.findtext(".//Journal/Title"),
                        "doi": doi, "title": "".join(art.find(".//ArticleTitle").itertext()),
                        "abstract": abstract})
        time.sleep(0.4)  # NCBI etiquette: <3 requests/s without an API key
    return out


def extract_metrics(papers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in papers.iterrows():
        for sent in re.split(r"(?<=[.;])\s+", p["abstract"] or ""):
            for m in METRIC_RE.finditer(sent):
                v = float(m["value"])
                if m["pct"] is None and v > 1:
                    if v > 100:
                        continue
                    v = v / 100 if v > 1.5 else v
                elif m["pct"]:
                    v = v / 100
                if not 0.4 <= v <= 1.0:
                    continue
                rows.append({"pmid": p["pmid"], "year": p["year"], "title": p["title"],
                             "metric": m["metric"].lower(), "value": round(v, 4), "sentence": sent.strip()})
    return pd.DataFrame(rows)


def main() -> None:
    hits: dict[str, set[str]] = {}
    for name, q in QUERIES.items():
        for pmid in esearch(q):
            hits.setdefault(pmid, set()).add(name)
        time.sleep(0.4)
    for pmid in KEY_PMIDS:
        hits.setdefault(pmid, set()).add("key_reference")
    papers = pd.DataFrame(efetch(sorted(hits)))
    papers["queries"] = papers["pmid"].map(lambda p: ";".join(sorted(hits[p])))
    papers["note"] = papers["pmid"].map(KEY_PMIDS)
    papers = papers.sort_values("year", ascending=False)
    papers.to_csv(TABLES / "literature.csv", index=False)
    metrics = extract_metrics(papers)
    metrics.to_csv(TABLES / "literature_metrics.csv", index=False)
    rows = []
    for pmid, study, metric, pattern in BASELINES:
        ab = papers.loc[papers["pmid"] == pmid, "abstract"]
        m = re.search(pattern, ab.iloc[0]) if len(ab) else None
        if m is None:
            print(f"WARNING: baseline {pmid} not found in scraped abstract")
            continue
        rows.append({"pmid": pmid, "study": study, "metric": metric, "value": float(m.group(1)) / 100,
                     "quote": m.group(0)})
    pd.DataFrame(rows).to_csv(TABLES / "baselines.csv", index=False)
    print(f"{len(papers)} papers scraped; {metrics['pmid'].nunique()} report numeric metrics "
          f"({len(metrics)} metric mentions)")
    print(papers["queries"].str.split(";").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
