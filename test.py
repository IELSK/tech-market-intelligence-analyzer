import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
br = df[df["Country"] == "Brazil"]
print(br.groupby("year")["LanguageHaveWorkedWith"].count())