import joblib
import pandas as pd
import sys
from pathlib import Path
from fastapi import FastAPI

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_ANALYSIS, DATA_PROCESSED, MODELS_DIR
from api.schemas import SalaryPredictionInput
from api.routers.languages import init_router as init_languages
from api.routers.trends import init_router as init_trends
from api.routers.countries import init_router as init_countries
from api.routers.prediction import init_router as init_prediction

# Load models and encoders
rf_model           = joblib.load(MODELS_DIR / "random_forest.pkl")
mlb_languages      = joblib.load(MODELS_DIR / "mlb_languages.pkl")
mlb_devtype        = joblib.load(MODELS_DIR / "mlb_devtype.pkl")
country_categories = joblib.load(MODELS_DIR / "country_categories.pkl")
feature_columns    = joblib.load(MODELS_DIR / "feature_columns.pkl")

# Load data 
df_processed      = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
language_metrics  = pd.read_parquet(DATA_ANALYSIS / "language_metrics.parquet")
opportunity_index = pd.read_parquet(DATA_ANALYSIS / "opportunity_index.parquet")
country_metrics   = pd.read_parquet(DATA_ANALYSIS / "country_language_metrics.parquet")

app = FastAPI(
    title="Tech Market Intelligence API",
    description="Market analysis and salary prediction for developers",
    version="1.0.0",
)

# Register routers 
app.include_router(init_languages(language_metrics, opportunity_index))
app.include_router(init_trends(opportunity_index, df_processed))
app.include_router(init_countries(country_metrics))
app.include_router(init_prediction(
    rf_model, mlb_languages, mlb_devtype,
    country_categories, feature_columns,
    language_metrics, opportunity_index,
))