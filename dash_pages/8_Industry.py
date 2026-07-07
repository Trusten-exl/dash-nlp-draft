import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("🏭 Industry Coverage")


st.markdown(
"""
Analyze which industries and economic sectors appear in newsroom coverage.
"""
)



industry = query("""
SELECT
    name,
    COUNT(*) AS stories
FROM article_sic
WHERE level='division'
GROUP BY name
ORDER BY stories DESC
""")


st.subheader(
    "Industry Coverage"
)


fig = px.bar(
    industry,
    x="name",
    y="stories"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader(
    "Industry Detail"
)


selected = st.selectbox(
    "Choose an industry",
    industry.name.tolist()
)



if selected:


    detail = query(f"""
    SELECT
        a.title,
        a.article_id,
        s.name,
        s.confidence

    FROM article_sic s

    JOIN articles a
    ON s.article_id=a.article_id

    WHERE s.name='{selected}'
    ORDER BY s.confidence DESC
    LIMIT 50
    """)


    st.dataframe(
        detail,
        use_container_width=True
    )