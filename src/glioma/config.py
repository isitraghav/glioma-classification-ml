"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

GDC_DATA = "https://api.gdc.cancer.gov/data/"
GDC_API = "https://api.gdc.cancer.gov/"

for _p in (RAW, INTERIM, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)
