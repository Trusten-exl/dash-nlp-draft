import streamlit as st

from pipeline.db.connection import query


st.title("🔎 Story Explorer")


search = st.text_input(
    "Search story titles"
)


if search:

    stories = query(f"""
    SELECT *
    FROM articles
    WHERE title LIKE '%{search}%'
    ORDER BY article_id DESC
    """)

else:

    stories = query("""
    SELECT *
    FROM articles
    ORDER BY article_id DESC
    LIMIT 100
    """)



if len(stories) == 0:
    st.warning("No stories found")
    st.stop()



article_id = st.selectbox(
    "Select story",
    stories.article_id
)



article = query(f"""
SELECT *
FROM articles
WHERE article_id={article_id}
""").iloc[0]



st.header(article.title)

st.caption(
    f"Word count: {article.word_count}"
)


st.write(article.text)



tabs = st.tabs(
[
"Topics",
"Intent",
"Entities",
"Sentiment",
"Political",
"Industry"
]
)



with tabs[0]:

    st.dataframe(
        query(f"""
        SELECT
            topic,
            confidence
        FROM article_topics
        WHERE article_id={article_id}
        ORDER BY rank
        """)
    )



with tabs[1]:

    st.dataframe(
        query(f"""
        SELECT
            intent,
            confidence
        FROM article_intents
        WHERE article_id={article_id}
        ORDER BY rank
        """)
    )



with tabs[2]:

    st.dataframe(
        query(f"""
        SELECT
            entity_text,
            entity_label,
            mention_count
        FROM entities
        WHERE article_id={article_id}
        ORDER BY mention_count DESC
        """)
    )



with tabs[3]:

    st.dataframe(
        query(f"""
        SELECT *
        FROM article_sentiment
        WHERE article_id={article_id}
        """)
    )



with tabs[4]:

    st.dataframe(
        query(f"""
        SELECT *
        FROM political_orientation
        WHERE article_id={article_id}

        UNION ALL

        SELECT *
        FROM political_salience
        WHERE article_id={article_id}
        """)
    )



with tabs[5]:

    st.dataframe(
        query(f"""
        SELECT
            level,
            name,
            confidence
        FROM article_sic
        WHERE article_id={article_id}
        """)
    )