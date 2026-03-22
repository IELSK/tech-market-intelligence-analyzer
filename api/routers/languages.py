import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()


def init_router(language_metrics: pd.DataFrame, opportunity_index: pd.DataFrame):
    @router.get("/top-languages")
    def top_languages(limit: int = 10):
        """Returns top languages ranked by popularity."""
        df = language_metrics.sort_values("popularity_pct", ascending=False).head(limit)
        return df.to_dict(orient="records")

    @router.get("/language/{name}")
    def language_detail(name: str):
        """Returns full details for a specific language."""
        lang_row = language_metrics[
            language_metrics["Language"].str.lower() == name.lower()
        ]
        if lang_row.empty:
            raise HTTPException(status_code=404, detail=f"Language '{name}' not found")

        result = lang_row.iloc[0].to_dict()

        opp_row = opportunity_index[
            opportunity_index["Language"].str.lower() == name.lower()
        ]
        if not opp_row.empty:
            result["growth_factor"]     = opp_row.iloc[0]["growth_factor"]
            result["opportunity_index"] = opp_row.iloc[0]["opportunity_index"]

        return result

    return router