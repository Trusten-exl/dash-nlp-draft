import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("✅ Editorial Audit")


st.markdown(
"""
Compare automated classification with manually reviewed editorial labels.
"""
)



comparison = query("""
SELECT

l.label_topic AS editorial_topic,

t.topic AS predicted_topic,

COUNT(*) AS stories

FROM article_labels l

JOIN article_topics t

ON l.article_id=t.article_id

WHERE t.rank=1

GROUP BY
editorial_topic,
predicted_topic

ORDER BY stories DESC
""")


st.subheader(
"Editorial Labels vs Automated Topics"
)


st.dataframe(
    comparison,
    use_container_width=True
)



agreement = (
    comparison.editorial_topic ==
    comparison.predicted_topic
).mean()



st.metric(
    "Topic Agreement",
    f"{agreement:.2%}"
)



st.divider()



st.subheader(
    "Most Common Disagreements"
)



errors = comparison[
    comparison.editorial_topic !=
    comparison.predicted_topic
]


fig = px.bar(
    errors.head(20),
    x="editorial_topic",
    y="stories",
    color="predicted_topic"
)


st.plotly_chart(
    fig,
    use_container_width=True
)