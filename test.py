import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
print(sorted(df["Country"].unique()))