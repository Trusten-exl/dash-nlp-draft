import streamlit as st
import plotly.express as px

from pipeline.db.connection import query


st.title("📄 Article Detail")


# -----------------------------
# Select Article
# -----------------------------

articles = query("""
SELECT
    article_id,
    title,
    word_count
FROM articles
ORDER BY article_id DESC
""")


selected_id = st.selectbox(
    "Select article",
    articles.article_id,
    format_func=lambda x:
        articles.loc[
            articles.article_id == x,
            "title"
        ].iloc[0]
)



# -----------------------------
# Article Information
# -----------------------------


article = query(f"""
SELECT *
FROM articles
WHERE article_id={selected_id}
""").iloc[0]



st.header(article.title)


c1,c2 = st.columns(2)


c1.metric(
    "Word Count",
    article.word_count
)


c2.write(
    f"Article ID: {article.article_id}"
)



st.divider()



with st.expander(
    "Article Text",
    expanded=True
):

    st.write(
        article.text
    )



# -----------------------------
# Editorial Summary
# -----------------------------

st.divider()

st.subheader(
    "Editorial Classification"
)



topics = query(f"""
SELECT
    topic,
    confidence
FROM article_topics

WHERE article_id={selected_id}

ORDER BY rank
""")


intents = query(f"""
SELECT
    intent,
    confidence
FROM article_intents

WHERE article_id={selected_id}

ORDER BY rank
""")


formats = query(f"""
SELECT
    format,
    confidence
FROM article_format

WHERE article_id={selected_id}

ORDER BY rank
""")


col1,col2,col3 = st.columns(3)


with col1:

    st.markdown("### Topics")

    st.dataframe(
        topics,
        hide_index=True
    )


with col2:

    st.markdown("### Intent")

    st.dataframe(
        intents,
        hide_index=True
    )


with col3:

    st.markdown("### Format")

    st.dataframe(
        formats,
        hide_index=True
    )



# -----------------------------
# Industry Classification
# -----------------------------

st.divider()

st.subheader(
    "Industry Coverage"
)


sic = query(f"""
SELECT
    level,
    code,
    name,
    confidence

FROM article_sic

WHERE article_id={selected_id}

ORDER BY confidence DESC
""")


st.dataframe(
    sic,
    hide_index=True,
    use_container_width=True
)



# -----------------------------
# Entities
# -----------------------------

st.divider()

st.subheader(
    "People, Organizations & Places"
)


entities = query(f"""
SELECT
    entity_text,
    entity_label,
    mention_count

FROM entities

WHERE article_id={selected_id}

ORDER BY mention_count DESC
""")


st.dataframe(
    entities,
    hide_index=True,
    use_container_width=True
)



# -----------------------------
# Entity Sentiment
# -----------------------------

st.subheader(
    "Entity Tone"
)


entity_sentiment = query(f"""
SELECT
    entity_text,
    sentiment,
    sentence_count

FROM entity_sentiment

WHERE article_id={selected_id}

ORDER BY sentence_count DESC
""")


if len(entity_sentiment):

    fig = px.bar(
        entity_sentiment,
        x="entity_text",
        y="sentiment",
        # size="sentence_count"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -----------------------------
# Article Sentiment
# -----------------------------


st.divider()

st.subheader(
    "Overall Story Tone"
)


sentiment = query(f"""
SELECT *

FROM article_sentiment

WHERE article_id={selected_id}
""")


if len(sentiment):

    st.dataframe(
        sentiment,
        hide_index=True
    )



# -----------------------------
# Political Analysis
# -----------------------------


st.divider()

st.subheader(
    "Political Analysis"
)


orientation = query(f"""
SELECT
    orientation,
    confidence

FROM political_orientation

WHERE article_id={selected_id}

ORDER BY rank
""")


salience = query(f"""
SELECT
    salience,
    confidence

FROM political_salience

WHERE article_id={selected_id}

ORDER BY rank
""")


col1,col2 = st.columns(2)


with col1:

    st.markdown(
        "### Orientation"
    )

    st.dataframe(
        orientation,
        hide_index=True
    )


with col2:

    st.markdown(
        "### Salience"
    )

    st.dataframe(
        salience,
        hide_index=True
    )



# -----------------------------
# Manual Label
# -----------------------------

st.divider()

st.subheader(
    "Editorial Ground Truth"
)


label = query(f"""
SELECT
    label_topic

FROM article_labels

WHERE article_id={selected_id}
""")


if len(label):

    st.success(
        f"Human Label: {label.iloc[0].label_topic}"
    )

else:

    st.info(
        "No manual editorial label available"
    )