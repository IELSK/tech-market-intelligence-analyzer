import plotly.express as px
import streamlit as st


def render_popularity(df_languages, rate, selected_currency):
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