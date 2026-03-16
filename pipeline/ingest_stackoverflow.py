import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RAW_DIR / "survey_results_public.csv"

df = pd.read_csv(CSV_FILE, low_memory=False)
print(f"Raw shape: {df.shape}")

COLUMNS = [
    "LanguageHaveWorkedWith",
    "ConvertedCompYearly",
    "YearsCode",
    "Country",
    "DevType",
]

df = df[COLUMNS]
print(f"Shape after selection: {df.shape}")

df = df.dropna(subset=["ConvertedCompYearly"])

df = df[df["ConvertedCompYearly"] > 1000]
df = df[df["ConvertedCompYearly"] < 500000]

df = df.dropna(subset=["LanguageHaveWorkedWith"])

print(f"Shape after cleaning: {df.shape}")

OUTPUT_FILE = PROCESSED_DIR / "dev_dataset.parquet"
df.to_parquet(OUTPUT_FILE, index=False)
print(f"Dataset saved to: {OUTPUT_FILE}")