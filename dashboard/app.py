import joblib
import pandas as pd
import requests
import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR

from tabs.popularity import render_popularity
from tabs.opportunity import render_opportunity
from tabs.trends import render_trends
from tabs.prediction import render_prediction
from tabs.countries import render_countries

load_dotenv()
API_URL              = os.getenv("API_URL", "http://localhost:8000")
EXCHANGE_RATE_API_URL = os.getenv("EXCHANGE_RATE_API_URL", "https://api.exchangerate-api.com/v4/latest/USD")

FEATURED_COUNTRIES_FLAGS = {
    "Brazil": "🇧🇷",
    "United States of America": "🇺🇸",
    "Canada": "🇨🇦",
    "United Kingdom of Great Britain and Northern Ireland": "🇬🇧",
    "Germany": "🇩🇪",
    "Australia": "🇦🇺",
    "Portugal": "🇵🇹",
    "Ireland": "🇮🇪",
    "Netherlands": "🇳🇱",
    "France": "🇫🇷",
    "Switzerland": "🇨🇭",
}

st.set_page_config(
    page_title="Tech Market Intelligence",
    page_icon="🧠",
    layout="wide",
)

# Fetch data
@st.cache_data(ttl=300)
def fetch_top_languages(limit: int = 30):
    response = requests.get(f"{API_URL}/top-languages?limit={limit}")
    return pd.DataFrame(response.json())

@st.cache_data(ttl=300)
def fetch_market_trends(limit: int = 30):
    response = requests.get(f"{API_URL}/market-trends?limit={limit}")
    return pd.DataFrame(response.json())

@st.cache_data(ttl=300)
def fetch_yearly_trends():
    response = requests.get(f"{API_URL}/yearly-trends")
    return pd.DataFrame(response.json())

@st.cache_data(ttl=300)
def fetch_country_analysis(country: str, limit: int = 15):
    response = requests.get(f"{API_URL}/country/{country}?limit={limit}")
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_exchange_rates():
    response = requests.get(EXCHANGE_RATE_API_URL)
    return response.json()["rates"]

@st.cache_data
def load_categories():
    countries = joblib.load(MODELS_DIR / "country_categories.pkl")
    mlb_dev   = joblib.load(MODELS_DIR / "mlb_devtype.pkl")
    mlb_lang  = joblib.load(MODELS_DIR / "mlb_languages.pkl")
    dev_types = sorted(mlb_dev.classes_.tolist())
    languages = sorted(mlb_lang.classes_.tolist())
    return countries, dev_types, languages

try:
    df_languages = fetch_top_languages()
    df_trends    = fetch_market_trends()
    df_yearly    = fetch_yearly_trends()
    rates        = fetch_exchange_rates()
except Exception:
    st.error("Could not connect to the API. Make sure it is running at http://localhost:8000")
    st.stop()

# Header
st.title("🧠 Tech Market Intelligence")
st.caption("Data sourced from Stack Overflow Developer Survey 2022–2025")

CURRENCIES = {
    "USD": 1.0,
    "BRL": rates.get("BRL", 5.0),
    "EUR": rates.get("EUR", 0.92),
    "GBP": rates.get("GBP", 0.79),
}
selected_currency = st.selectbox("Currency", options=list(CURRENCIES.keys()), index=0)
rate = CURRENCIES[selected_currency]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Popularity vs Salary",
    "🏆 Opportunity Ranking",
    "📈 Yearly Trends",
    "💰 Salary Prediction",
    "🌍 Country Analysis",
])

with tab1:
    render_popularity(df_languages, rate, selected_currency)

with tab2:
    render_opportunity(df_trends, rate, selected_currency)

with tab3:
    render_trends(df_yearly, df_languages)

with tab4:
    render_prediction(rate, selected_currency, API_URL, load_categories)

with tab5:
    render_countries(rate, selected_currency, API_URL, FEATURED_COUNTRIES_FLAGS, fetch_country_analysis)