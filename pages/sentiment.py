import streamlit as st
import plotly.express as px
from pipeline.db.connection import query

st.title("🙂 Sentiment Analysis")

articles = query("SELECT * FROM articles")
sent = query("SELECT * FROM article_sentiment")
topics = query("SELECT * FROM article_topics WHERE rank = 1")

df = articles.merge(sent, on="article_id")
df = df.merge(topics, on="article_id")

fig = px.histogram(df, x="sentiment_score")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sentiment by Topic")

topic_sent = df.groupby("topic")["sentiment_score"].mean().reset_index()

fig2 = px.bar(topic_sent, x="topic", y="sentiment_score")
st.plotly_chart(fig2, use_container_width=True)