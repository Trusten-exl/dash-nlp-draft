import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("🎭 Tone & Sentiment")


st.markdown(
"""
Understand the emotional tone of published coverage.
"""
)


sentiment = query("""
SELECT *
FROM article_sentiment
""")


positive = sentiment.positive.sum()
neutral = sentiment.neutral.sum()
negative = sentiment.negative.sum()



c1,c2,c3 = st.columns(3)


c1.metric(
    "Positive Sentences",
    int(positive)
)

c2.metric(
    "Neutral Sentences",
    int(neutral)
)

c3.metric(
    "Negative Sentences",
    int(negative)
)



st.divider()



st.subheader(
    "Overall Coverage Tone"
)


tone = query("""
SELECT
    'Positive' AS sentiment,
    SUM(positive) AS count
FROM article_sentiment

UNION ALL

SELECT
    'Neutral',
    SUM(neutral)
FROM article_sentiment

UNION ALL

SELECT
    'Negative',
    SUM(negative)
FROM article_sentiment
""")


fig = px.pie(
    tone,
    names="sentiment",
    values="count"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader(
    "Sentiment Distribution"
)


fig = px.histogram(
    sentiment,
    x="sentiment_score",
    nbins=40
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader(
    "Entity Tone"
)


entity_sentiment = query("""
SELECT
    entity_text,
    AVG(sentiment) AS avg_sentiment,
    SUM(sentence_count) AS sentences
FROM entity_sentiment
GROUP BY entity_text
ORDER BY sentences DESC
LIMIT 50
""")


st.dataframe(
    entity_sentiment,
    use_container_width=True
)