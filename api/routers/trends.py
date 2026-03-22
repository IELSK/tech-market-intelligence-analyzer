import pandas as pd
from fastapi import APIRouter

router = APIRouter()


def init_router(opportunity_index: pd.DataFrame, df_processed: pd.DataFrame):
    @router.get("/market-trends")
    def market_trends(limit: int = 10):
        """Returns languages ranked by opportunity index."""
        df = opportunity_index.sort_values("opportunity_index", ascending=False).head(limit)
        return df.to_dict(orient="records")

    @router.get("/yearly-trends")
    def yearly_trends():
        """Returns language popularity per year for trend analysis."""
        total_per_year = df_processed.groupby("year")["LanguageHaveWorkedWith"].count()

        df_exploded = (
            df_processed.assign(Language=df_processed["LanguageHaveWorkedWith"].str.split(";"))
            .explode("Language")
            .reset_index(drop=True)
        )
        df_exploded["Language"] = df_exploded["Language"].str.strip()

        lang_year = (
            df_exploded.groupby(["year", "Language"])["ConvertedCompYearly"]
            .count()
            .reset_index(name="count")
        )
        lang_year["popularity"] = lang_year.apply(
            lambda row: row["count"] / total_per_year[row["year"]] * 100, axis=1
        )

        return lang_year[["year", "Language", "popularity"]].to_dict(orient="records")

    return router