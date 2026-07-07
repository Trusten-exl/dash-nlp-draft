import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("🏛 Political Coverage")


st.markdown(
"""
Understand the political profile of newsroom coverage.
"""
)



orientation = query("""
SELECT
    orientation,
    COUNT(*) AS stories
FROM political_orientation
WHERE rank=1
GROUP BY orientation
""")


st.subheader(
    "Coverage Orientation"
)


fig = px.pie(
    orientation,
    names="orientation",
    values="stories"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



salience = query("""
SELECT
    salience,
    COUNT(*) AS stories
FROM political_salience
WHERE rank=1
GROUP BY salience
""")


st.subheader(
    "Political Salience"
)


fig = px.bar(
    salience,
    x="salience",
    y="stories"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.divider()



st.subheader(
    "Political Coverage by Topic"
)


topic_political = query("""
SELECT
    t.topic,
    p.orientation,
    COUNT(*) AS stories

FROM article_topics t

JOIN political_orientation p
ON t.article_id=p.article_id

WHERE t.rank=1
AND p.rank=1

GROUP BY
    t.topic,
    p.orientation
""")


st.dataframe(
    topic_political,
    use_container_width=True
)