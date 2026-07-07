import streamlit as st
import plotly.express as px
from pipeline.db.connection import query

st.title("📊 Topic Analytics")

topics = query("SELECT * FROM article_topics WHERE rank = 1")

topic_counts = topics["topic"].value_counts().reset_index()
topic_counts.columns = ["topic", "count"]

fig = px.bar(topic_counts, x="topic", y="count")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Confidence by Topic")

conf = topics.groupby("topic")["confidence"].mean().reset_index()

fig2 = px.bar(conf, x="topic", y="confidence")
st.plotly_chart(fig2, use_container_width=True)