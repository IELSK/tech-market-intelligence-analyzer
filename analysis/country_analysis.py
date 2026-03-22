import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, DATA_ANALYSIS, FEATURED_COUNTRIES

DATA_ANALYSIS.mkdir(parents=True, exist_ok=True)


# Load
df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
print(f"Loaded dataset: {df.shape}")

# Filter featured countries
df = df[df["Country"].isin(FEATURED_COUNTRIES)]
print(f"Shape after country filter: {df.shape}")

# Explode languages
df_exploded = (
    df.assign(Language=df["LanguageHaveWorkedWith"].str.split(";"))
    .explode("Language")
    .reset_index(drop=True)
)
df_exploded["Language"] = df_exploded["Language"].str.strip()

# Metrics per country
total_per_country = df.groupby("Country")["LanguageHaveWorkedWith"].count()

lang_country_counts = (
    df_exploded.groupby(["Country", "Language"])["ConvertedCompYearly"]
    .count()
    .reset_index(name="count")
)

lang_country_counts["popularity"] = lang_country_counts.apply(
    lambda row: row["count"] / total_per_country[row["Country"]] * 100, axis=1
)

lang_country_metrics = (
    df_exploded.groupby(["Country", "Language"])
    .agg(
        developer_count=("ConvertedCompYearly", "count"),
        mean_salary=("ConvertedCompYearly", "mean"),
        median_salary=("ConvertedCompYearly", "median"),
    )
    .reset_index()
)

lang_country_metrics = lang_country_metrics.merge(
    lang_country_counts[["Country", "Language", "popularity"]],
    on=["Country", "Language"],
    how="left",
)

lang_country_metrics["mean_salary"]   = lang_country_metrics["mean_salary"].round(2)
lang_country_metrics["median_salary"] = lang_country_metrics["median_salary"].round(2)
lang_country_metrics["popularity"]    = lang_country_metrics["popularity"].round(2)

# CAGR per country 
total_per_country_year = df.groupby(["Country", "year"])["LanguageHaveWorkedWith"].count()

lang_country_year = (
    df_exploded.groupby(["Country", "year", "Language"])["ConvertedCompYearly"]
    .count()
    .reset_index(name="count")
)

lang_country_year["popularity"] = lang_country_year.apply(
    lambda row: row["count"] / total_per_country_year[(row["Country"], row["year"])] * 100, axis=1
)

popularity_pivot = lang_country_year.pivot_table(
    index=["Country", "Language"], columns="year", values="popularity"
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

lang_country_metrics = lang_country_metrics.merge(
    popularity_pivot[["Country", "Language", "growth_factor"]],
    on=["Country", "Language"],
    how="left",
)

# Opportunity index per country
global_median_salary = df["ConvertedCompYearly"].median()

MIN_POPULARITY  = 5.0
MIN_COUNT_PCT   = 0.03  # language must represent at least 3% of country respondents

lang_country_metrics = lang_country_metrics.merge(
    total_per_country.reset_index(name="total_respondents"),
    on="Country",
    how="left",
)

lang_country_metrics["min_count"] = (lang_country_metrics["total_respondents"] * MIN_COUNT_PCT).astype(int)
lang_country_metrics = lang_country_metrics[lang_country_metrics["developer_count"] >= lang_country_metrics["min_count"]]
lang_country_metrics = lang_country_metrics[lang_country_metrics["popularity"] >= MIN_POPULARITY]
lang_country_metrics = lang_country_metrics.dropna(subset=["growth_factor"])
lang_country_metrics = lang_country_metrics.drop(columns=["min_count", "total_respondents"])

lang_country_metrics["opportunity_index"] = (
    (lang_country_metrics["median_salary"] / global_median_salary)
    * np.log1p(lang_country_metrics["popularity"])
    * np.log1p(lang_country_metrics["growth_factor"])
).round(4)

# Save
lang_country_metrics.to_parquet(DATA_ANALYSIS / "country_language_metrics.parquet", index=False)
print(f"Saved to: {DATA_ANALYSIS / 'country_language_metrics.parquet'}")

# Preview 
print("\nTop 5 opportunities in Brazil:")
us = lang_country_metrics[lang_country_metrics["Country"] == "Brazil"]
print(us.sort_values("opportunity_index", ascending=False).head(5)[
    ["Language", "popularity", "median_salary", "growth_factor", "opportunity_index"]
].to_string(index=False))