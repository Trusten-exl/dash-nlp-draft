import streamlit as st
from pipeline.db.connection import query

st.title("🔍 Article Detail View")

article_id = st.number_input("Enter Article ID", min_value=1, step=1)

if article_id:

    article = query(
        "SELECT * FROM articles WHERE article_id = ?",
        [article_id]
    )

    topics = query(
        "SELECT * FROM article_topics WHERE article_id = ? ORDER BY rank",
        [article_id]
    )

    industry = query(
        "SELECT * FROM article_sic WHERE article_id = ?",
        [article_id]
    )

    intent = query(
        "SELECT * FROM article_intents WHERE article_id = ? ORDER BY rank",
        [article_id]
    )

    format = query(
        "SELECT * FROM article_format WHERE article_id = ? ORDER BY rank",
        [article_id]
    )

    sent = query(
        "SELECT * FROM article_sentiment WHERE article_id = ?",
        [article_id]
    )

    ents = query(
        "SELECT * FROM entities WHERE article_id = ? ORDER BY mention_count DESC",
        [article_id]
    )

    ent_sent = query(
        "SELECT * FROM entity_sentiment WHERE article_id = ?",
        [article_id]
    )

    article_label = query(
        "SELECT * FROM article_labels WHERE article_id = ?",
        [article_id]
    )

    if len(article) == 0:
        st.error("Article not found")
        st.stop()

    st.subheader(article["title"].iloc[0])

    st.write(article["text"].iloc[0])

    st.markdown("### Topics")
    st.dataframe(topics)

    if industry.empty == False:
        st.markdown('### Industry')
        st.dataframe(industry)

    st.markdown('### Intent')
    st.dataframe(intent)

    st.markdown('### Format')
    st.dataframe(format)

    st.markdown('### Ground Truth')
   
    if len(article_label) == 0:
        st.write('Not Labeled')
    else:
        st.dataframe(article_label)

    st.markdown("### Sentiment")
    st.dataframe(sent)

    st.markdown("### Entities")
    st.dataframe(ents)

    st.markdown("### Entity Sentiment")
    st.dataframe(ent_sent)