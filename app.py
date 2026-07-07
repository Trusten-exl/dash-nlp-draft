# app.py
# file for running streamlit web app
# connect to pages

import streamlit as st
from pipeline.db.connection import query

st.set_page_config(page_title="News NLP Dashboard", layout="wide")

st.title("📰 News NLP Analytics Dashboard")

articles = query("SELECT * FROM articles")
topics = query("SELECT * FROM article_topics WHERE rank = 1")
sent = query("SELECT * FROM article_sentiment")

col1, col2, col3 = st.columns(3)

col1.metric("Articles", len(articles))
col2.metric("Topics", topics["topic"].nunique())
col3.metric("Avg Sentiment", round(sent["sentiment_score"].mean(), 3))

st.markdown("### Quick Start")
st.write("Use the sidebar pages to explore articles, topics, sentiment, and entities.")