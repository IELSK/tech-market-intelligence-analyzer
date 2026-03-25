import plotly.express as px
import streamlit as st


def render_opportunity(df_trends, rate, selected_currency):
    st.subheader("Language Opportunity Ranking")
    st.caption("Based on salary, market presence and growth factor")

    limit = st.slider("Number of languages", min_value=5, max_value=20, value=10)

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