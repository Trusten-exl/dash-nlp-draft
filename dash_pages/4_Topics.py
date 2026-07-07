import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("🗂 Topic Landscape")


topics = query("""
SELECT
    topic,
    COUNT(*) AS stories,
    AVG(confidence) AS avg_confidence
FROM article_topics
WHERE rank=1
GROUP BY topic
ORDER BY stories DESC
""")


st.dataframe(
    topics,
    use_container_width=True
)



fig = px.scatter(
    topics,
    x="avg_confidence",
    y="stories",
    size="stories",
    text="topic"
)


st.plotly_chart(
    fig,
    use_container_width=True
)