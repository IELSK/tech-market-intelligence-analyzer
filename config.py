from pathlib import Path

ROOT = Path(__file__).parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ANALYSIS  = ROOT / "data" / "analysis"
MODELS_DIR     = ROOT / "models" / "artifacts"
SURVEY_YEARS   = [2022, 2023, 2024, 2025]