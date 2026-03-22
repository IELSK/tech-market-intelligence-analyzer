import pandas as pd
import requests
import streamlit as st


def render_prediction(rate, selected_currency, api_url, load_categories):
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
                response = requests.post(f"{api_url}/salary-prediction", json=payload)

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