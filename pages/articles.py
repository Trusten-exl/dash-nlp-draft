import streamlit as st
from pipeline.db.connection import query

st.title("📄 Articles Explorer")

articles = query("SELECT * FROM articles")
topics = query("SELECT * FROM article_topics WHERE rank = 1")
sent = query("SELECT * FROM article_sentiment")

df = articles.merge(topics, on="article_id", how="left")
df = df.merge(sent, on="article_id", how="left")

# Filters
topic_filter = st.multiselect("Topic", df["topic"].dropna().unique())
search = st.text_input("Search title")

filtered = df.copy()

if topic_filter:
    filtered = filtered[filtered["topic"].isin(topic_filter)]

if search:
    filtered = filtered[df["title"].str.contains(search, case=False, na=False)]

st.dataframe(
    filtered[
        ["article_id", "title", "topic", "confidence", "sentiment_score"]
    ],
    use_container_width=True
)

# Click-through hint
st.info("Go to 'Article Detail' page and enter an article_id for full view.")