import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, DATA_ANALYSIS

DATA_ANALYSIS.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
language_metrics = pd.read_parquet(DATA_ANALYSIS / "language_metrics.parquet")
print(f"Loaded dataset: {df.shape}")

# Explode languages
df_exploded = (
    df.assign(Language=df["LanguageHaveWorkedWith"].str.split(";"))
    .explode("Language")
    .reset_index(drop=True)
)
df_exploded["Language"] = df_exploded["Language"].str.strip()

# Popularity per year 
total_per_year = df.groupby("year")["LanguageHaveWorkedWith"].count()

lang_year_counts = (
    df_exploded.groupby(["year", "Language"])["ConvertedCompYearly"]
    .count()
    .reset_index(name="count")
)

lang_year_counts["popularity"] = lang_year_counts.apply(
    lambda row: row["count"] / total_per_year[row["year"]] * 100, axis=1
)

# Filter languages with enough data
MIN_COUNT_PER_YEAR = 300

valid_languages = (
    lang_year_counts[lang_year_counts["count"] >= MIN_COUNT_PER_YEAR]
    .groupby("Language")["year"]
    .nunique()
)
valid_languages = valid_languages[valid_languages == 4].index
lang_year_counts = lang_year_counts[lang_year_counts["Language"].isin(valid_languages)]
print(f"Languages with enough data for CAGR: {len(valid_languages)}")

# CAGR
popularity_pivot = lang_year_counts.pivot(
    index="Language", columns="year", values="popularity"
).reset_index()

start_year = 2022
end_year   = 2025
n_years    = end_year - start_year

def calculate_cagr(row):
    start = row.get(start_year)
    end   = row.get(end_year)

    if pd.isna(start) or pd.isna(end) or start == 0:
        return np.nan

    return (end / start) ** (1 / n_years) - 1

popularity_pivot["growth_factor"] = popularity_pivot.apply(calculate_cagr, axis=1)

# Global median salary
global_median_salary = df["ConvertedCompYearly"].median()
print(f"Global median salary: ${global_median_salary:,.0f}")

# Merge and calculate opportunity index
metrics = language_metrics.merge(
    popularity_pivot[["Language", "growth_factor"]],
    on="Language",
    how="left",
)

metrics = metrics[metrics["developer_count"] >= 100]
metrics = metrics.dropna(subset=["growth_factor"])

MIN_POPULARITY = 5.0
metrics = metrics[metrics["popularity_pct"] >= MIN_POPULARITY]

metrics["opportunity_index"] = (
    (metrics["median_salary"] / global_median_salary)
    * np.log1p(metrics["popularity_pct"])
    * np.log1p(metrics["growth_factor"])
).round(4)

print("\nTop 10 opportunities:")
print(
    metrics[["Language", "popularity_pct", "median_salary", "growth_factor", "opportunity_index"]]
    .head(10)
    .to_string(index=False)
)

# Save
metrics.to_parquet(DATA_ANALYSIS / "opportunity_index.parquet", index=False)
print(f"\nSaved to: {DATA_ANALYSIS / 'opportunity_index.parquet'}")