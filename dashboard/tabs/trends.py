import plotly.express as px
import streamlit as st


def render_trends(df_yearly, df_languages):
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