import streamlit as st
import plotly.express as px
from pipeline.db.connection import query

st.title("👥 Entity Analytics")

entities = query("SELECT * FROM entities")
ent_sent = query("SELECT * FROM entity_sentiment")

st.subheader("Top Entities")

top = (
    entities.groupby("entity_text")["mention_count"]
    .sum()
    .reset_index()
    .sort_values("mention_count", ascending=False)
    .head(20)
)

fig = px.bar(top, x="entity_text", y="mention_count")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Entity Sentiment Landscape")

summary = (
    ent_sent.groupby("entity_text")
    .agg(
        avg_sentiment=("sentiment", "mean"),
        mentions=("sentence_count", "sum")
    )
    .reset_index()
)

fig2 = px.scatter(
    summary,
    x="avg_sentiment",
    y="mentions",
    hover_name="entity_text"
)

st.plotly_chart(fig2, use_container_width=True)