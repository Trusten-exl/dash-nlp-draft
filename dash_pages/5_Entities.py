import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("👥 People & Organizations")


st.markdown(
"""
Understand which people, organizations, and places dominate newsroom coverage.
"""
)


entities = query("""
SELECT
    entity_text,
    entity_label,
    SUM(mention_count) AS mentions
FROM entities
GROUP BY entity_text, entity_label
ORDER BY mentions DESC
LIMIT 100
""")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Most Mentioned Entities")

    st.dataframe(
        entities,
        use_container_width=True
    )


with col2:

    st.subheader("Entity Types")


    entity_types = query("""
    SELECT
        entity_label,
        COUNT(*) AS count
    FROM entities
    GROUP BY entity_label
    ORDER BY count DESC
    """)


    fig = px.pie(
        entity_types,
        names="entity_label",
        values="count"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



st.divider()


st.subheader(
    "Entity Coverage Search"
)


entity_search = st.text_input(
    "Search for an entity"
)


if entity_search:


    results = query(f"""
    SELECT
        entity_text,
        entity_label,
        mention_count,
        article_id
    FROM entities
    WHERE entity_text LIKE '%{entity_search}%'
    ORDER BY mention_count DESC
    """)


    st.dataframe(
        results,
        use_container_width=True
    )