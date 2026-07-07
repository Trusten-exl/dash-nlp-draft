import streamlit as st


st.set_page_config(
    page_title="Newsroom Intelligence",
    page_icon="📰",
    layout="wide"
)


overview = st.Page(
    "dash_pages/1_Overview.py",
    title="Overview",
    icon="📰"
)

explorer = st.Page(
    "dash_pages/2_Story_Explorer.py",
    title="Story Explorer",
    icon="🔎"
)

trends = st.Page(
    "dash_pages/3_Coverage_Trends.py",
    title="Coverage Trends",
    icon="📈"
)

topics = st.Page(
    "dash_pages/4_Topics.py",
    title="Topics",
    icon="🗂"
)

entities = st.Page(
    "dash_pages/5_Entities.py",
    title="Entities",
    icon="👥"
)

sentiment = st.Page(
    "dash_pages/6_Sentiment.py",
    title="Sentiment",
    icon="🎭"
)

political = st.Page(
    "dash_pages/7_Political.py",
    title="Political",
    icon="🏛"
)

industry = st.Page(
    "dash_pages/8_Industry.py",
    title="Industry",
    icon="🏭"
)

audit = st.Page(
    "dash_pages/9_Editorial_Audit.py",
    title="Editorial Audit",
    icon="✅"
)

detail = st.Page(
    "dash_pages/10_Article_Detail.py",
    title="Article Detail",
    icon="📄"
)


pg = st.navigation(
    {
        "Newsroom": [
            overview,
            detail,
            explorer,
            trends
        ],

        "Analysis": [
            topics,
            entities,
            sentiment,
            political,
            industry
        ],

        "Quality": [
            audit
        ]
    }
)


pg.run()