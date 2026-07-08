import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline.db.connection import query


# ============================================================
# Helper Functions
# ============================================================

def confidence_bar(df, label_col, value_col):

    if df.empty:
        st.info("No data available")
        return

    fig = px.bar(
        df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        height=350,
        showlegend=False,
        xaxis_title="Confidence",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def metric_card(label, value):

    st.metric(
        label,
        value
    )


def section_header(title, subtitle=None):

    st.subheader(title)

    if subtitle:
        st.caption(subtitle)



# ============================================================
# Main Page
# ============================================================

def article_detail_page(article_id):


    # --------------------------------------------------------
    # Load Article
    # --------------------------------------------------------

    article_df = query(
        """
        SELECT *
        FROM articles
        WHERE article_id = ?
        """,
        (article_id,)
    )


    if article_df.empty:
        st.error("Article not found")
        return


    article = article_df.iloc[0]



    # --------------------------------------------------------
    # Hero Section
    # --------------------------------------------------------

    if article["image_url"]:

        try:

            st.image(
                article["image_url"],
                caption="Extracted Article Image",
                use_container_width=True
            )

        except:
            pass



    st.markdown(
        f"""
        <h1 style="
            font-size:38px;
            line-height:1.15;
        ">
            {article['title']}
        </h1>
        """,
        unsafe_allow_html=True
    )


    meta = []

    if article["sitename"]:
        meta.append(article["sitename"])

    if article["author"]:
        meta.append(
            f"By {article['author']}"
        )

    if article["publish_date"]:
        meta.append(article["publish_date"])


    st.caption(
        " • ".join(meta)
    )


    if article["url"]:

        st.markdown(
            f"""
            <a href="{article['url']}" target="_blank">
            🔗 Open Original Article
            </a>
            """,
            unsafe_allow_html=True
        )



    # --------------------------------------------------------
    # Metadata Cards
    # --------------------------------------------------------

    section_header(
        "Article Overview"
    )


    c1,c2,c3,c4 = st.columns(4)


    with c1:
        metric_card(
            "Words",
            f"{article['word_count']:,}"
            if article["word_count"]
            else "-"
        )


    with c2:
        metric_card(
            "Language",
            article["language"]
            if article["language"]
            else "-"
        )


    with c3:
        metric_card(
            "Publisher",
            article["hostname"]
            if article["hostname"]
            else "-"
        )


    with c4:
        metric_card(
            "Modified",
            article["modified_date"]
            if article["modified_date"]
            else "-"
        )



    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if article["description"]:

        section_header(
            "Summary"
        )

        st.info(
            article["description"]
        )



    # --------------------------------------------------------
    # Topic + Intent Analysis
    # --------------------------------------------------------

    section_header(
        "Content Classification",
        "Model confidence scores"
    )


    col1,col2 = st.columns(2)



    # Topics
    with col1:

        st.subheader(
            "Topics"
        )

        topics = query(
            """
            SELECT topic, confidence
            FROM article_topics
            WHERE article_id = ?
            ORDER BY confidence DESC
            """,
            (article_id,)
        )


        if not topics.empty:

            confidence_bar(
                topics,
                "topic",
                "confidence"
            )

        else:

            st.info(
                "No topics detected"
            )



    # Intent
    with col2:

        st.subheader(
            "Article Format"
        )


        formats = query(
            """
            SELECT format, confidence
            FROM article_format
            WHERE article_id = ?
            ORDER BY confidence DESC
            """,
            (article_id,)
        )


        if not formats.empty:

            confidence_bar(
                formats,
                "format",
                "confidence"
            )

        else:

            st.info(
                "No format classification"
            )



    # --------------------------------------------------------
    # Sentiment Overview
    # --------------------------------------------------------

    section_header(
        "Article Sentiment"
    )


    sentiment = query(
        """
        SELECT *
        FROM article_sentiment
        WHERE article_id = ?
        """,
        (article_id,)
    )


    if not sentiment.empty:


        s = sentiment.iloc[0]


        sent_df = pd.DataFrame(
            {
                "Sentiment":[
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                "Count":[
                    s["positive"],
                    s["neutral"],
                    s["negative"]
                ]
            }
        )


        fig = px.pie(
            sent_df,
            names="Sentiment",
            values="Count",
            hole=.55
        )


        fig.update_layout(
            height=350
        )


        col1,col2 = st.columns([1,1])


        with col1:

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            metric_card(
                "Overall Sentiment Score",
                round(
                    s["sentiment_score"],
                    3
                )
            )


    # --------------------------------------------------------
    # SIC / Industry Classification
    # --------------------------------------------------------

    section_header(
        "Industry Classification",
        "Detected SIC categories"
    )


    sic = query(
        """
        SELECT level,
               code,
               name,
               confidence
        FROM article_sic
        WHERE article_id = ?
        ORDER BY confidence DESC
        """,
        (article_id,)
    )


    if not sic.empty:


        sic_display = sic.copy()


        sic_display["classification"] = (
            sic_display["level"].astype(str)
            + " - "
            + sic_display["name"]
        )


        fig = px.bar(
            sic_display,
            x="confidence",
            y="classification",
            orientation="h",
            text="confidence"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )


        fig.update_layout(
            height=350,
            xaxis_title="Confidence",
            yaxis_title=""
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.info(
            "No industry classification available"
        )



    # --------------------------------------------------------
    # Political Analysis
    # --------------------------------------------------------

    section_header(
        "Political Analysis"
    )


    col1,col2 = st.columns(2)



    with col1:

        st.subheader(
            "Political Salience"
        )


        salience = query(
            """
            SELECT salience,
                   confidence
            FROM political_salience
            WHERE article_id = ?
            ORDER BY confidence DESC
            """,
            (article_id,)
        )


        if not salience.empty:


            fig = px.bar(
                salience,
                x="salience",
                y="confidence",
                text="confidence"
            )


            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )


            fig.update_layout(
                height=300
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No political salience detected"
            )



    with col2:

        st.subheader(
            "Political Orientation"
        )


        orientation = query(
            """
            SELECT orientation,
                   confidence
            FROM political_orientation
            WHERE article_id = ?
            ORDER BY confidence DESC
            """,
            (article_id,)
        )


        if not orientation.empty:


            fig = px.bar(
                orientation,
                x="orientation",
                y="confidence",
                text="confidence"
            )


            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )


            fig.update_layout(
                height=300
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        else:

            st.info(
                "No political orientation detected"
            )



    # --------------------------------------------------------
    # Entities
    # --------------------------------------------------------

    section_header(
        "Named Entities",
        "Extracted people, organizations, locations, and other entities"
    )


    entities = query(
        """
        SELECT entity_text,
               entity_label,
               mention_count
        FROM entities
        WHERE article_id = ?
        ORDER BY mention_count DESC
        """,
        (article_id,)
    )


    if not entities.empty:


        entity_counts = (
            entities
            .groupby("entity_label")
            .size()
            .reset_index(
                name="count"
            )
        )


        fig = px.bar(
            entity_counts,
            x="entity_label",
            y="count",
            text="count"
        )


        fig.update_layout(
            height=300
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



        st.subheader(
            "Entity Details"
        )


        for label, group in entities.groupby(
            "entity_label"
        ):

            with st.expander(
                f"{label} ({len(group)})"
            ):

                st.dataframe(
                    group[
                        [
                            "entity_text",
                            "mention_count"
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True
                )


    else:

        st.info(
            "No entities extracted"
        )



    # --------------------------------------------------------
    # Entity Sentiment
    # --------------------------------------------------------

    section_header(
        "Entity Sentiment",
        "Sentiment surrounding extracted entities"
    )


    entity_sent = query(
        """
        SELECT entity_text,
               sentiment,
               sentence_count
        FROM entity_sentiment
        WHERE article_id = ?
        ORDER BY sentence_count DESC
        """,
        (article_id,)
    )


    if not entity_sent.empty:


        fig = px.scatter(
            entity_sent,
            x="sentence_count",
            y="sentiment",
            size="sentence_count",
            hover_name="entity_text"
        )


        fig.update_layout(
            height=400,
            xaxis_title="Mentions",
            yaxis_title="Sentiment"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            entity_sent,
            hide_index=True,
            use_container_width=True
        )


    else:

        st.info(
            "No entity sentiment available"
        )



    # --------------------------------------------------------
    # Article Text
    # --------------------------------------------------------

    section_header(
        "Full Article"
    )


    with st.expander(
        "Read Article Text"
    ):

        st.markdown(
            article["text"]
        )



# ============================================================
# Streamlit Entry Point
# ============================================================

if __name__ == "__main__":

    st.set_page_config(
        page_title="Article Detail",
        layout="wide"
    )


    # Load available articles
    articles = query(
        """
        SELECT article_id,
               title,
               sitename,
               publish_date
        FROM articles
        ORDER BY publish_date DESC
        """
    )


    if articles.empty:

        st.error(
            "No articles found"
        )

    else:

        # Create readable selector labels
        articles["display"] = (
            articles["title"].fillna("Untitled")
            + " | "
            + articles["sitename"].fillna("")
            + " | "
            + articles["publish_date"].fillna("")
        )


        selected = st.selectbox(
            "Select Article",
            articles["display"]
        )


        selected_id = articles.loc[
            articles["display"] == selected,
            "article_id"
        ].iloc[0]


        article_detail_page(
            int(selected_id)
        )