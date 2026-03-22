import plotly.express as px
import streamlit as st


def render_countries(rate, selected_currency, api_url, featured_countries, fetch_country_analysis):
    st.subheader("Country Analysis")
    st.caption("Top language opportunities by country")

    country_options = [f"{flag} {name}" for name, flag in featured_countries.items()]
    selected_option = st.selectbox("Select a country", options=country_options)
    selected_country_name = selected_option.split(" ", 1)[1]

    limit_country = st.slider("Number of languages", min_value=5, max_value=20, value=10, key="country_limit")

    df_country = fetch_country_analysis(selected_country_name, limit=limit_country)

    if not df_country.empty:
        df_country_plot = df_country.copy()
        df_country_plot["median_salary"] = (df_country_plot["median_salary"] * rate).round(2)
        df_country_plot["mean_salary"]   = (df_country_plot["mean_salary"] * rate).round(2)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Opportunity Index**")
            df_sorted = df_country_plot.sort_values("opportunity_index")
            fig = px.bar(
                df_sorted,
                x="opportunity_index",
                y="Language",
                orientation="h",
                color="opportunity_index",
                color_continuous_scale="Teal",
                labels={"opportunity_index": "Opportunity Index"},
                height=450,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**Popularity vs Median Salary**")
            fig = px.scatter(
                df_country_plot,
                x="popularity",
                y="median_salary",
                size="developer_count",
                text="Language",
                labels={
                    "popularity": "Popularity (%)",
                    "median_salary": f"Median Salary ({selected_currency})",
                },
                height=450,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            df_country_plot[[
                "Language", "popularity", "median_salary", "growth_factor", "opportunity_index"
            ]].reset_index(drop=True),
            width="stretch",
        )
    else:
        st.warning("No data available for this country.")