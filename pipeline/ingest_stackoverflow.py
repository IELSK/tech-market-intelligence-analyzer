import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
CSV_FILE = DATA_RAW / "survey_results_public.csv"

df = pd.read_csv(CSV_FILE, low_memory=False)
print(f"Raw shape: {df.shape}")

# ── Select relevant columns ───────────────────────────────────────────────────
COLUMNS = [
    "LanguageHaveWorkedWith",
    "ConvertedCompYearly",
    "YearsCode",
    "Country",
    "DevType",
]

df = df[COLUMNS]
print(f"Shape after column selection: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df.dropna(subset=["ConvertedCompYearly"])
df = df[df["ConvertedCompYearly"] > 1000]
df = df[df["ConvertedCompYearly"] < 500000]
df = df.dropna(subset=["LanguageHaveWorkedWith"])
print(f"Shape after cleaning: {df.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_FILE = DATA_PROCESSED / "dev_dataset.parquet"
df.to_parquet(OUTPUT_FILE, index=False)
print(f"Dataset saved to: {OUTPUT_FILE}")