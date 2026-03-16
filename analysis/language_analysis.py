import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, DATA_ANALYSIS

DATA_ANALYSIS.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
print(f"Loaded dataset: {df.shape}")

# Explode languages
df_exploded = (
    df.assign(Language=df["LanguageHaveWorkedWith"].str.split(";"))
    .explode("Language")
    .reset_index(drop=True)
)
df_exploded["Language"] = df_exploded["Language"].str.strip()
print(f"Exploded dataset: {df_exploded.shape}")

# Metrics
total_devs = df["LanguageHaveWorkedWith"].count()

language_metrics = (
    df_exploded.groupby("Language")
    .agg(
        developer_count=("ConvertedCompYearly", "count"),
        mean_salary=("ConvertedCompYearly", "mean"),
        median_salary=("ConvertedCompYearly", "median"),
    )
    .reset_index()
)

language_metrics["popularity_pct"] = (
    language_metrics["developer_count"] / total_devs * 100
).round(2)

language_metrics["mean_salary"] = language_metrics["mean_salary"].round(2)
language_metrics["median_salary"] = language_metrics["median_salary"].round(2)

# Filter and sort
language_metrics = language_metrics[language_metrics["developer_count"] >= 100]
language_metrics = language_metrics.sort_values("popularity_pct", ascending=False)

print(language_metrics.head(10).to_string(index=False))

# Save
language_metrics.to_parquet(DATA_ANALYSIS / "language_metrics.parquet", index=False)
print(f"Saved to: {DATA_ANALYSIS / 'language_metrics.parquet'}")