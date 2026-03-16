import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RAW_DIR / "survey_results_public.csv"

df = pd.read_csv(CSV_FILE, low_memory=False)
print(f"Shape bruto: {df.shape}")