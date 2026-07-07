import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("📰 Newsroom Overview")


articles = query("""
SELECT *
FROM articles
""")


sentiment = query("""
SELECT *
FROM article_sentiment
""")


topics = query("""
SELECT
    topic,
    COUNT(*) AS stories
FROM article_topics
WHERE rank = 1
GROUP BY topic
ORDER BY stories DESC
LIMIT 15
""")


formats = query("""
SELECT
    format,
    COUNT(*) AS stories
FROM article_format
WHERE rank = 1
GROUP BY format
""")


# Metrics

c1,c2,c3,c4 = st.columns(4)


c1.metric(
    "Total Stories",
    len(articles)
)

c2.metric(
    "Average Story Length",
    round(articles.word_count.mean())
)

c3.metric(
    "Total Words",
    f"{articles.word_count.sum():,}"
)

c4.metric(
    "Negative Sentences",
    int(sentiment.negative.sum())
)



st.divider()



st.subheader("Most Covered Topics")


fig = px.bar(
    topics,
    x="topic",
    y="stories"
)

st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader("Story Formats")


fig = px.pie(
    formats,
    names="format",
    values="stories"
)

st.plotly_chart(
    fig,
    use_container_width=True
)