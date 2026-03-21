import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_ANALYSIS, MODELS_DIR

# Load models and encoders
rf_model           = joblib.load(MODELS_DIR / "random_forest.pkl")
mlb_languages      = joblib.load(MODELS_DIR / "mlb_languages.pkl")
mlb_devtype        = joblib.load(MODELS_DIR / "mlb_devtype.pkl")
country_categories = joblib.load(MODELS_DIR / "country_categories.pkl")
feature_columns    = joblib.load(MODELS_DIR / "feature_columns.pkl")

# Load analysis data
language_metrics  = pd.read_parquet(DATA_ANALYSIS / "language_metrics.parquet")
opportunity_index = pd.read_parquet(DATA_ANALYSIS / "opportunity_index.parquet")

app = FastAPI(
    title="Tech Market Intelligence API",
    description="Market analysis and salary prediction for developers",
    version="1.0.0",
)

# Schemas
class SalaryPredictionInput(BaseModel):
    languages: list[str]
    years_of_experience: float
    country: str
    dev_type: str

# Routes
@app.get("/top-languages")
def top_languages(limit: int = 10):
    """Returns top languages ranked by popularity."""
    df = language_metrics.sort_values("popularity_pct", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/market-trends")
def market_trends(limit: int = 10):
    """Returns languages ranked by opportunity index."""
    df = opportunity_index.sort_values("opportunity_index", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/language/{name}")
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


@app.post("/salary-prediction")
def salary_prediction(data: SalaryPredictionInput):
    """Predicts salary based on languages, experience, country and dev type."""

    # Encode languages
    lang_encoded = pd.DataFrame(
        mlb_languages.transform([data.languages]),
        columns=mlb_languages.classes_,
    )

    # Encode DevType
    try:
        devtype_encoded = pd.DataFrame(
            mlb_devtype.transform([[data.dev_type]]),
            columns=mlb_devtype.classes_,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown dev_type: '{data.dev_type}'")

    # Encode Country
    if data.country not in country_categories:
        raise HTTPException(status_code=400, detail=f"Unknown country: '{data.country}'")
    country_encoded = country_categories.index(data.country)

    # Build feature row
    input_df = pd.concat(
        [
            pd.DataFrame([[data.years_of_experience, country_encoded]], columns=["YearsCode", "Country_encoded"]),
            devtype_encoded.reset_index(drop=True),
            lang_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    # Ensure column order matches training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict 
    predicted_salary = float(rf_model.predict(input_df)[0])

    # Enrich response with market data
    lang_data = []
    for lang in data.languages:
        row = language_metrics[language_metrics["Language"].str.lower() == lang.lower()]
        if not row.empty:
            entry = row.iloc[0].to_dict()
            opp = opportunity_index[opportunity_index["Language"].str.lower() == lang.lower()]
            if not opp.empty:
                entry["growth_factor"]     = opp.iloc[0]["growth_factor"]
                entry["opportunity_index"] = opp.iloc[0]["opportunity_index"]
            lang_data.append(entry)

    return {
        "predicted_salary": round(predicted_salary, 2),
        "input": data.model_dump(),
        "language_market_data": lang_data,
    }