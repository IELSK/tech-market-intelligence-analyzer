import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
print(df[["YearsCode", "DevType", "Country"]].isna().sum())
print(f"Total rows: {len(df)}")