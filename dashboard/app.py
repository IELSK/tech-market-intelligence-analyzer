import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR

API_URL = "http://localhost:8000"

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

@st.cache_data(ttl=3600)
def fetch_exchange_rates():
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Popularity vs Salary",
    "🏆 Opportunity Ranking",
    "📈 Yearly Trends",
    "💰 Salary Prediction",
])

# Tab 1: Popularity vs Salary
with tab1:
    st.subheader("Popularity vs Median Salary")
    st.caption("Bubble size represents number of developers")

    df_plot = df_languages.copy()
    df_plot["median_salary"] = (df_plot["median_salary"] * rate).round(2)
    df_plot["mean_salary"]   = (df_plot["mean_salary"] * rate).round(2)

    fig = px.scatter(
        df_plot,
        x="popularity_pct",
        y="median_salary",
        size="developer_count",
        text="Language",
        labels={
            "popularity_pct": "Popularity (%)",
            "median_salary": f"Median Salary ({selected_currency})",
        },
        height=600,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Tab 2: Opportunity Ranking
with tab2:
    st.subheader("Language Opportunity Ranking")
    st.caption("Based on salary, market presence and growth factor")

    limit = st.slider("Number of languages", min_value=5, max_value=30, value=15)

    df_top = df_trends.head(limit).copy()
    df_top["median_salary"] = (df_top["median_salary"] * rate).round(2)
    df_top_sorted = df_top.sort_values("opportunity_index")

    fig = px.bar(
        df_top_sorted,
        x="opportunity_index",
        y="Language",
        orientation="h",
        color="opportunity_index",
        color_continuous_scale="Teal",
        labels={"opportunity_index": "Opportunity Index"},
        height=600,
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        df_top[[
            "Language", "popularity_pct", "median_salary", "growth_factor", "opportunity_index"
        ]].reset_index(drop=True),
        width="stretch",
    )

# Tab 3: Yearly Trends
with tab3:
    st.subheader("Popularity Trend by Year")
    st.caption("Evolution of language adoption from 2022 to 2025")

    top_languages = df_languages["Language"].head(15).tolist()
    selected = st.multiselect(
        "Select languages",
        options=sorted(df_yearly["Language"].unique().tolist()),
        default=top_languages[:5],
    )

    if selected:
        df_filtered = df_yearly[df_yearly["Language"].isin(selected)]
        fig = px.line(
            df_filtered,
            x="year",
            y="popularity",
            color="Language",
            markers=True,
            labels={
                "year": "Year",
                "popularity": "Popularity (%)",
            },
            height=500,
        )
        fig.update_xaxes(tickvals=[2022, 2023, 2024, 2025])
        st.plotly_chart(fig, width="stretch")

# Tab 4: Salary Prediction
with tab4:
    st.subheader("Salary Prediction")
    st.caption("Predict your market salary based on your profile")

    countries, dev_types, languages = load_categories()

    col1, col2 = st.columns(2)

    with col1:
        selected_languages = st.multiselect("Languages", options=languages, default=["Python"])
        selected_country   = st.selectbox("Country", options=countries)
        selected_devtype   = st.selectbox("Dev Type", options=dev_types)
        years_exp          = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

    with col2:
        if st.button("Predict Salary", type="primary"):
            if not selected_languages:
                st.warning("Select at least one language.")
            else:
                payload = {
                    "languages": selected_languages,
                    "years_of_experience": years_exp,
                    "country": selected_country,
                    "dev_type": selected_devtype,
                }
                response = requests.post(f"{API_URL}/salary-prediction", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    predicted_converted = data["predicted_salary"] * rate
                    st.metric("Predicted Salary", f"{selected_currency} {predicted_converted:,.0f}")

                    st.subheader("Language Market Data")
                    df_lang_data = pd.DataFrame(data["language_market_data"])
                    if not df_lang_data.empty:
                        df_lang_data["median_salary"] = (df_lang_data["median_salary"] * rate).round(2)
                        df_lang_data["mean_salary"]   = (df_lang_data["mean_salary"] * rate).round(2)
                        st.dataframe(
                            df_lang_data[[
                                "Language", "popularity_pct", "median_salary",
                                "growth_factor", "opportunity_index"
                            ]].reset_index(drop=True),
                            width="stretch",
                        )
                else:
                    st.error(f"Prediction error: {response.json().get('detail')}")