import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, SURVEY_YEARS

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "LanguageHaveWorkedWith",
    "ConvertedCompYearly",
    "YearsCode",
    "Country",
    "DevType",
]

# Load
frames = []

for year in SURVEY_YEARS:
    csv_file = DATA_RAW / str(year) / "survey_results_public.csv"
    df_year = pd.read_csv(csv_file, low_memory=False)
    print(f"{year} raw shape: {df_year.shape}")
    df_year = df_year[COLUMNS]
    df_year["year"] = year
    frames.append(df_year)

df = pd.concat(frames, ignore_index=True)
print(f"Combined loaded shape: {df.shape}")

# Clean
df = df.dropna(subset=["ConvertedCompYearly"])
df = df[df["ConvertedCompYearly"] > 1000]
df = df[df["ConvertedCompYearly"] < 500000]
df = df.dropna(subset=["LanguageHaveWorkedWith"])
df["YearsCode"] = pd.to_numeric(df["YearsCode"], errors="coerce")
print(f"Shape after cleaning: {df.shape}")

# Save
output_file = DATA_PROCESSED / "dev_dataset.parquet"
df.to_parquet(output_file, index=False)
print(f"Dataset saved to: {output_file}")