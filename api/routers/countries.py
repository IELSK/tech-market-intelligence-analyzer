import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()


def init_router(country_metrics: pd.DataFrame):
    @router.get("/country/{name}")
    def country_analysis(name: str, limit: int = 10):
        """Returns top language opportunities for a specific country."""
        df = country_metrics[
            country_metrics["Country"].str.lower() == name.lower()
        ]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Country '{name}' not found")

        df_sorted = df.sort_values("opportunity_index", ascending=False).head(limit)
        return df_sorted.to_dict(orient="records")

    return router