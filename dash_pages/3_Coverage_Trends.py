import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("📈 Coverage Trends")


topics = query("""
SELECT
    topic,
    COUNT(*) stories
FROM article_topics
WHERE rank=1
GROUP BY topic
ORDER BY stories DESC
""")


st.subheader(
    "Coverage Distribution"
)


fig = px.bar(
    topics,
    x="topic",
    y="stories"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader(
    "Story Length Distribution"
)


lengths = query("""
SELECT word_count
FROM articles
""")


fig = px.histogram(
    lengths,
    x="word_count",
    nbins=40
)


st.plotly_chart(
    fig,
    use_container_width=True
)