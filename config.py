from pathlib import Path

ROOT = Path(__file__).parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ANALYSIS  = ROOT / "data" / "analysis"
MODELS_DIR     = ROOT / "models" / "artifacts"
SURVEY_YEARS   = [2022, 2023, 2024, 2025]
FEATURED_COUNTRIES = [
    "Brazil",
    "United States of America",
    "Canada",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "Australia",
    "Portugal",
    "Ireland",
    "Netherlands",
    "France",
    "Switzerland",
]
