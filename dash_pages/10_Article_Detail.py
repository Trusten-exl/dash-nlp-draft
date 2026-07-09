# ============================================================
# article_detail.py
# Part 1 / 4
# Imports, configuration, data loading, helpers
# ============================================================

from __future__ import annotations

import math
import html
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.db.connection import query


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Article Intelligence",
    page_icon="📰",
    layout="wide",
)

if "show_reader" not in st.session_state:
    st.session_state.show_reader = False

# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLDS = {
    "topic": 0.60,
    "format": 0.30,
    "sic": 0.20,
    "orientation": 0.150,
    "salience": 0.0,
}
MAX_HEATMAP_ENTITIES = 20

CARD_BORDER = "1px solid rgba(180,180,180,.25)"
CARD_RADIUS = "12px"
CARD_PADDING = "0.75rem"

ENTITY_LABELS = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "PRODUCT": "Products",
    "EVENT": "Events",
    "LAW": "Laws",
    "WORK_OF_ART": "Works",
    "NORP": "Groups",
    "FAC": "Facilities",
    "LANGUAGE": "Languages",
}

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container{
    padding-top:5rem;
    padding-bottom:1rem;
    max-width:1700px;
}

.metric-container{
    padding:0;
}

.intel-card{
    border:1px solid rgba(120,120,120,.25);
    border-radius:12px;
    padding:14px;
    background:rgba(255,255,255,.02);
    height:115px;
}

.intel-title{
    font-size:.78rem;
    color:#888;
    text-transform:uppercase;
    letter-spacing:.05rem;
}

.intel-value{
    font-size:1.2rem;
    font-weight:700;
    margin-top:4px;
}

.badge{
    display:inline-block;
    padding:4px 10px;
    border-radius:20px;
    font-size:.72rem;
    font-weight:600;
    margin-top:10px;
}

.hero-img img{
    border-radius:12px;
}

.entity-chip{
    display:inline-block;
    margin:4px;
    padding:6px 10px;
    border-radius:18px;
    border:1px solid rgba(160,160,160,.25);
    font-size:.80rem;
}

.section-title{
    font-size:1.2rem;
    font-weight:700;
    margin-bottom:.4rem;
}

.small-muted{
    color:#888;
    font-size:.85rem;
}

[data-testid="stAppViewContainer"]{
    overflow-y:auto;
}

[data-testid="stHeader"]{
    z-index:999;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Utility Functions
# ============================================================


def confidence_badge(conf: float) -> str:
    if conf >= 0.90:
        color = "#1e8e3e"
    elif conf >= 0.75:
        color = "#2d9cdb"
    elif conf >= 0.60:
        color = "#f4b400"
    else:
        color = "#db4437"

    return (
        f"<span class='badge' "
        f"style='background:{color}22;color:{color};'>"
        f"{conf:.0%}"
        f"</span>"
    )


def sentiment_color(score: float) -> str:
    if score >= 0.1:
        return "#1b9e4b"

    if score <= -0.1:
        return "#d93025"

    return "#888888"


def sentiment_highlight_text(df):

    if df is None or df.empty:
        return None


    html_parts = []


    for _, row in df.iterrows():

        sentence = html.escape(
            str(row["sentence"])
        )

        sentiment = str(
            row["sentiment"]
        ).lower()

        score = float(
            row["score"]
        )


        if sentiment == "positive":

            background = (
                f"rgba(46,204,113,{(abs(score) ** 0.7) * 0.35})"
            )

        elif sentiment == "negative":

            background = (
                f"rgba(231,76,60,{(abs(score) ** 0.7) * 0.35})"
            )

        else:

            background = "transparent"


        html_parts.append(
            f"""
            <span
            style="
                background-color:{background};
                padding:2px 4px;
                border-radius:4px;
                border-bottom:1px solid rgba(0,0,0,.05);
            "
            title="Sentiment: {sentiment} | Score: {score:.2f}">
            {sentence}
            </span>
            """
        )


    return " ".join(html_parts)


def orientation_color(label: str) -> str:
    label = str(label).lower()

    if "left" in label:
        return "#2d6cdf"

    if "right" in label:
        return "#d93025"

    return "#7a7a7a"


def salience_color(label: str) -> str:
    label = str(label).lower()

    if "low" in label:
        return "#2e8b57"

    if "medium" in label:
        return "#f9ab00"

    return "#d93025"


def parse_date(value):

    if value is None:
        return ""

    if value == "":
        return ""

    try:
        return pd.to_datetime(value).strftime("%b %d, %Y")
    except Exception:
        return str(value)


# ============================================================
# Database Loaders
# ============================================================



def load_articles():

    return query(
        """
        SELECT *
        FROM articles
        ORDER BY publish_date DESC
        """
    )



def load_article(article_id):

    df = query(
        """
        SELECT *
        FROM articles
        WHERE article_id = ?
        """,
        (article_id,),
    )

    if df.empty:
        return None

    return df.iloc[0]



def load_topics(article_id):

    return query(
        """
        SELECT *
        FROM article_topics
        WHERE article_id = ?
        ORDER BY rank
        """,
        (article_id,),
    )



def load_formats(article_id):

    return query(
        """
        SELECT *
        FROM article_format
        WHERE article_id = ?
        ORDER BY rank
        """,
        (article_id,),
    )


# @st.cache_data(show_spinner=False)
def load_sic(article_id):

    return query(
        """
        SELECT *
        FROM article_sic
        WHERE article_id = ?
        """,
        (article_id,),
    )


# @st.cache_data(show_spinner=False)
def load_sentiment(article_id):

    df = query(
        """
        SELECT *
        FROM article_sentiment
        WHERE article_id = ?
        """,
        (article_id,),
    )

    if df.empty:
        return None

    return df.iloc[0]

def load_sentence_sentiment(article_id):

    return query(
        """
        SELECT
            sentence,
            sentiment,
            score
        FROM sent_sentiment
        WHERE article_id = ?
        ORDER BY id
        """,
        (article_id,)
    )

# @st.cache_data(show_spinner=False)
def load_orientation(article_id):

    return query(
        """
        SELECT *
        FROM political_orientation
        WHERE article_id = ?
        ORDER BY rank
        """,
        (article_id,),
    )


# @st.cache_data(show_spinner=False)
def load_salience(article_id):

    return query(
        """
        SELECT *
        FROM political_salience
        WHERE article_id = ?
        ORDER BY rank
        """,
        (article_id,),
    )


# @st.cache_data(show_spinner=False)
def load_entities(article_id):

    return query(
        """
        SELECT *
        FROM entities
        WHERE article_id = ?
        ORDER BY mention_count DESC
        """,
        (article_id,),
    )


# @st.cache_data(show_spinner=False)
def load_entity_sentiment(article_id):

    return query(
        """
        SELECT *
        FROM entity_sentiment
        WHERE article_id = ?
        """,
        (article_id,),
    )

@st.dialog("Highlighted Article")
def show_article_dialog():

    st.components.v1.html(
        article_html,
        height=1000,
        scrolling=True,
    )


# ============================================================
# Helper Selection Functions
# ============================================================


def best_prediction(df, value_col, threshold):

    if df is None or df.empty:
        return None


    df = df[
        df["confidence"] >= threshold
    ]


    if df.empty:
        return None


    return (
        df
        .sort_values(
            "confidence",
            ascending=False
        )
        .iloc[0]
    )


def best_sic(df, level):

    if df.empty:
        return None

    tmp = df[df["level"] == level]

    if tmp.empty:
        return None

    tmp = tmp.sort_values("confidence", ascending=False)

    row = tmp.iloc[0]

    if row["confidence"] < CONFIDENCE_THRESHOLDS["sic"]:
        return None

    return row


# ============================================================
# Sidebar
# ============================================================

articles = load_articles()

if articles.empty:
    st.error("No articles found.")
    st.stop()

article_lookup = {
    f"{row['title'][:60]} | {row['sitename']} | ID:{row['article_id']}": row['article_id']
    for _, row in articles.iterrows()
}

st.sidebar.write(
    "Articles loaded:",
    len(articles)
)

# ============================================================
# Article Selector
# ============================================================

st.markdown(
    """
    <div class='section-title'>
    Article Selection
    </div>
    """,
    unsafe_allow_html=True
)



selection = st.selectbox(
    "Choose Article",
    list(article_lookup.keys()),
    label_visibility="collapsed"
    )


ARTICLE_ID = article_lookup[selection]

article = load_article(ARTICLE_ID)

topics = load_topics(ARTICLE_ID)
formats = load_formats(ARTICLE_ID)
sic = load_sic(ARTICLE_ID)
sentiment = load_sentiment(ARTICLE_ID)
orientation = load_orientation(ARTICLE_ID)
salience = load_salience(ARTICLE_ID)
entities = load_entities(ARTICLE_ID)
entity_sent = load_entity_sentiment(ARTICLE_ID)
sentence_sent = load_sentence_sentiment(ARTICLE_ID)
highlighted_text = sentiment_highlight_text(sentence_sent)

article_html = f"""
<div style="
    font-family: Georgia, serif;
    font-size:1.05rem;
    line-height:1.9;
    padding:20px;
">
    {highlighted_text}
</div>
"""


    # ============================================================
    # Hero Section
    # ============================================================
if st.session_state.show_reader:

    dashboard_col, reader_col = st.columns(
        [3,2],
        gap="large"
    )
    with dashboard_col: 
            hero_left, hero_right = st.columns(
                [1.15, 3],
                gap="large"
            )


            with hero_left:

                if article["image_url"]:

                    st.markdown(
                        "<div class='hero-img'>",
                        unsafe_allow_html=True
                    )

                    st.image(
                        article["image_url"],
                        width=260
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                else:

                    st.info("No article image available")


            with hero_right:

                st.markdown(
                    f"""
                    <div style="
                        font-size:1.55rem;
                        font-weight:700;
                        line-height:1.2;
                        margin-bottom:.25rem;
                    ">
                    {article['title']}
                    </div>

                    <div style="
                        font-size:.85rem;
                        color:#888;
                    ">
                    {article['sitename'] or article['hostname']}
                    &nbsp; • &nbsp;
                    {parse_date(article['publish_date'])}
                    </div>
                    """,
                    unsafe_allow_html=True
                    )


                if article["description"]:

                    st.markdown(
                    f"""
                    <div style="
                    font-size:.9rem;
                    color:#555;
                    max-height:55px;
                    overflow:hidden;
                    ">
                    {article['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                    )


                meta_cols = st.columns(4)


                with meta_cols[0]:

                    if article["author"]:

                        st.markdown(
                            f"""
                            **Author**

                            {article['author']}
                            """
                        )


                with meta_cols[1]:

                    st.markdown(
                        f"""
                        **Language**

                        {article['language'] or "Unknown"}
                        """
                    )


                with meta_cols[2]:

                    if article["url"]:

                        st.link_button(
                            "Open Original Article",
                            article["url"]
                        )

                with meta_cols[3]:
                    if st.button("📖 Read Article"):
                        st.session_state.show_reader = not st.session_state.show_reader



            # ============================================================
            # Metadata Strip
            # ============================================================

            topic_summary = "N/A"


            if topics is not None and not topics.empty:

                topic_rows = topics[
                    topics["confidence"] >= CONFIDENCE_THRESHOLDS["topic"]
                ].sort_values(
                    "confidence",
                    ascending=False
                )


                if not topic_rows.empty:

                    topic_summary = " • ".join(
                        topic_rows["topic"].tolist()
                    )

            format_summary = "N/A"


            if formats is not None and not formats.empty:

                format_rows = formats.sort_values(
                    "confidence",
                    ascending=False
                )


                format_summary = format_rows.iloc[0]["format"]

            st.divider()


            meta = [
                ("Words", f"{article['word_count']:,}"),
                ("Source", article["sitename"]),
                # ("Language", article["language"]),
                ("Published", parse_date(article["publish_date"])),
                ("Modified", parse_date(article["modified_date"])),
                ("Topics", topic_summary),
                ("Format", format_summary)
            ]


            cols = st.columns(6)


            for col, (label,value) in zip(cols,meta):

                with col:

                    st.markdown(
                    f"""
                    <div style="
                        border:1px solid rgba(120,120,120,.2);
                        border-radius:8px;
                        padding:8px;
                        height:55px;
                    ">

                    <div style="
                        font-size:.65rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    {label}
                    </div>

                    <div style="
                        font-size:.95rem;
                        font-weight:600;
                    ">
                    {value}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                    )

            # ============================================================
            # Intelligence Summary
            # ============================================================


            st.divider()

            # st.markdown(
            #     """
            #     <div class='section-title'>
            #     Intelligence Summary
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )



            def render_intel_card(
                    title,
                    value,
                    color="#888"
            ):

                st.markdown(
                    f"""
                    <div style="
                        border:1px solid rgba(120,120,120,.25);
                        border-left:5px solid {color};
                        border-radius:12px;
                        padding:12px;
                        min-height:100px;
                    ">

                    <div style="
                        font-size:.7rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    {title}
                    </div>


                    <div style="
                        font-size:1rem;
                        font-weight:700;
                        margin-top:8px;
                        color:{color};
                    ">
                    {value}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )



            # ============================================================
            # Build Intelligence Objects
            # ============================================================

            intel_cards = []


            # # Topics

            # if topics is not None and not topics.empty:

            #     topic_rows = (
            #         topics[
            #             topics["confidence"] >= CONFIDENCE_THRESHOLDS["topic"]
            #         ]
            #         .sort_values(
            #             "confidence",
            #             ascending=False
            #         )
            #     )


            #     if not topic_rows.empty:

            #         intel_cards.append(
            #             (
            #                 "Topics",
            #                 " • ".join(
            #                     topic_rows["topic"].tolist()
            #                 ),
            #                 "#2d6cdf"
            #             )
            #         )



            # # Article Format

            # fmt = best_prediction(
            #     formats,
            #     "format",
            #     CONFIDENCE_THRESHOLDS["format"]
            # )

            # if fmt is not None:

            #     intel_cards.append(
            #         (
            #             "Format",
            #             fmt["format"],
            #             "#6f42c1"
            #         )
            #     )


            # Sentiment

            # if sentiment is not None:

            #     score = sentiment["sentiment_score"]

            #     label = (
            #         "Positive"
            #         if score > .15
            #         else
            #         "Negative"
            #         if score < -.15
            #         else
            #         "Neutral"
            #     )

            #     intel_cards.append(
            #         (
            #             "Overall Sentiment",
            #             label,
            #             None,
            #             sentiment_color(score)
            #         )
            #     )



            # ============================================================
            # Render Cards
            # ============================================================

            # if intel_cards:


            #     card_cols = st.columns(
            #         4
            #     )


            #     for index, card in enumerate(
            #         intel_cards
            #     ):

            #         with card_cols[
            #             index % 4
            #         ]:

            #             render_intel_card(
            #                 card[0],
            #                 card[1],
            #                 card[2],
            #             )

            # else:

            #     st.info(
            #         "No classifications exceeded the confidence threshold."
            #     )


            # ============================================================
            # Sentiment Summary Panel
            # ============================================================

            if sentiment is not None:

                st.markdown(
                    """
                    <div class='section-title'
                    style='margin-top:1rem'>
                    Article Sentiment
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if sentiment is not None:

                    score = sentiment["sentiment_score"]


                    if score > 0.15:
                        sentiment_label = "Positive"

                    elif score < -0.15:
                        sentiment_label = "Negative"

                    else:
                        sentiment_label = "Neutral"


                    st.markdown(
                        f"""
                        <div style="
                            font-size:1.2rem;
                            font-weight:700;
                            margin-bottom:15px;
                        ">
                        Overall Sentiment:
                        <span style="
                            color:{sentiment_color(score)};
                        ">
                        {sentiment_label}
                        </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                sent_cols = st.columns(5)


                sentiment_metrics = [

                    (
                        "Score",
                        round(
                            sentiment["sentiment_score"],
                            3
                        )
                    ),

                    (
                        "Positive",
                        int(sentiment["positive"])
                    ),

                    (
                        "Neutral",
                        int(sentiment["neutral"])
                    ),

                    (
                        "Negative",
                        int(sentiment["negative"])
                    )

                ]


                for col, metric in zip(
                    sent_cols,
                    sentiment_metrics
                ):

                    with col:

                        st.markdown(
                            f"""
                            <div style="
                                border:1px solid rgba(120,120,120,.25);
                                border-radius:10px;
                                padding:10px;
                            ">

                            <div style="
                                color:#888;
                                font-size:.75rem;
                            ">
                            {metric[0]}
                            </div>

                            <div style="
                                font-size:1.2rem;
                                font-weight:700;
                            ">
                            {metric[1]}
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            # ============================================================
            # Entity Sentiment Treemap
            # ============================================================


            st.markdown(
                """
                <div class='section-title'>
                Entity Intelligence Map
                </div>
                """,
                unsafe_allow_html=True
            )


            if (
                entity_sent is not None
                and not entity_sent.empty
            ):


                entity_map = entity_sent.copy()


                # Add mention counts

                if (
                    entities is not None
                    and not entities.empty
                ):

                    entity_map = entity_map.merge(
                        entities[
                            [
                                "entity_text",
                                "entity_label",
                                "mention_count"
                            ]
                        ],
                        on="entity_text",
                        how="left"
                    )


                else:

                    entity_map["mention_count"] = 1

                    entity_map["entity_label"] = "Other"



                # Convert sentiment numeric

                entity_map["sentiment"] = pd.to_numeric(
                    entity_map["sentiment"],
                    errors="coerce"
                )


                # Aggregate entities

                entity_summary = (

                    entity_map
                    .groupby(
                        [
                            "entity_text",
                            "entity_label"
                        ],
                        as_index=False
                    )
                    .agg(
                        {
                            "sentiment":"mean",
                            "mention_count":"max"
                        }
                    )

                )


                # Remove missing values

                entity_summary = (
                    entity_summary
                    .dropna(
                        subset=[
                            "entity_text"
                        ]
                    )
                )


                # Limit largest entities

                entity_summary = (
                    entity_summary
                    .sort_values(
                        "mention_count",
                        ascending=False
                    )
                    .head(40)
                )


                if not entity_summary.empty:


                    fig = px.treemap(

                        entity_summary,

                        path=[
                            "entity_label",
                            "entity_text"
                        ],

                        values="mention_count",

                        color="sentiment",

                        color_continuous_scale=[
                            "red",
                            "white",
                            "green"
                        ],

                        range_color=[
                            -1,
                            1
                        ],

                    )


                    fig.update_traces(

                        texttemplate=
                        "<b>%{label}</b><br>%{value} mentions",

                        hovertemplate=
                        """
                        <b>%{label}</b><br>
                        Mentions: %{value}<br>
                        Sentiment: %{color:.2f}
                        <extra></extra>
                        """

                    )


                    fig.update_layout(

                        height=300,

                        margin=dict(
                            t=10,
                            l=5,
                            r=5,
                            b=5
                        ),

                        coloraxis_colorbar=dict(
                            title="Sentiment"
                        )

                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                else:

                    st.info(
                        "No entity sentiment available."
                    )


            else:

                st.info(
                    "No entity sentiment data available."
                )

            # ============================================================
            # Industry + Political Intelligence
            # ============================================================

            industry_col, political_col = st.columns(
                [1,1],
                gap="large"
            )


            with industry_col:

                # ============================================================
                # SIC Hierarchy
                # ============================================================


                def get_sic(level):

                    if sic is None or sic.empty:
                        return None

                    row = (
                        sic[
                            sic["level"] == level
                        ]
                        .sort_values(
                            "confidence",
                            ascending=False
                        )
                    )

                    if row.empty:
                        return None

                    return row.iloc[0]["name"]



                sic_division = get_sic("division")
                sic2 = get_sic("sic2")
                sic3 = get_sic("sic3")
                sic4 = get_sic("sic4")



                if any(
                    [
                        sic_division,
                        sic2,
                        sic3,
                        sic4
                    ]
                ):


                    st.markdown(
                        """
                        <div class='section-title'>
                        Industry Classification
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.markdown(

                        f"""
                        <div style="
                            border-radius:14px;
                            border:1px solid rgba(120,120,120,.25);
                            padding:16px;
                            background:rgba(0,0,0,.02);
                        ">


                        <div style="
                            font-size:.75rem;
                            color:#888;
                            text-transform:uppercase;
                        ">
                        SIC Division
                        </div>


                        <div style="
                            font-size:1.35rem;
                            font-weight:800;
                            color:#1f77b4;
                            margin-bottom:14px;
                        ">
                        {sic_division or "N/A"}
                        </div>



                        <div style="
                            font-size:.7rem;
                            color:#888;
                            text-transform:uppercase;
                        ">
                        SIC 2
                        </div>


                        <div style="
                            font-size:1.1rem;
                            font-weight:700;
                            color:#2ca02c;
                            margin-bottom:12px;
                        ">
                        {sic2 or "N/A"}
                        </div>



                        <div style="
                            font-size:.65rem;
                            color:#888;
                            text-transform:uppercase;
                        ">
                        SIC 3
                        </div>


                        <div style="
                            font-size:.95rem;
                            font-weight:600;
                            color:#9467bd;
                            margin-bottom:10px;
                        ">
                        {sic3 or "N/A"}
                        </div>



                        <div style="
                            font-size:.6rem;
                            color:#888;
                            text-transform:uppercase;
                        ">
                        SIC 4
                        </div>


                        <div style="
                            font-size:.85rem;
                            color:#d62728;
                            font-weight:500;
                        ">
                        {sic4 or "N/A"}
                        </div>


                        </div>
                        """,

                        unsafe_allow_html=True

                    )


            # ============================================================
            # Political Intelligence
            # ============================================================

            with political_col:

                st.markdown("### Political Intelligence")

                orientation_row = best_prediction(
                    orientation,
                    "orientation",
                    CONFIDENCE_THRESHOLDS["orientation"]
                )

                salience_row = best_prediction(
                    salience,
                    "salience",
                    CONFIDENCE_THRESHOLDS["salience"]
                )

                if orientation_row is not None:

                    orientation_value = orientation_row["orientation"]

                    orientation_map = {
                        "Progressive or left-wing": 10,
                        "Center-left": 30,
                        "Centrist or politically neutral": 50,
                        "Center-right": 70,
                        "Right-wing or conservative": 90,
                    }

                    orientation_score = orientation_map.get(
                        orientation_value,
                        50
                    )

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge",

                            value=orientation_score,

                            gauge={

                                "shape": "angular",

                                "axis": {
                                    "range": [0, 100],
                                    "tickvals": [0, 25, 50, 75, 100],
                                    "ticktext": [
                                        "Left",
                                        "Center<br>Left",
                                        "Neutral",
                                        "Center<br>Right",
                                        "Right",
                                    ],
                                },

                                "bar": {
                                    "color": "#333333",
                                    "thickness": 0.20,
                                },

                                "steps": [

                                    {
                                        "range": [0, 20],
                                        "color": "#2F6BFF",
                                    },

                                    {
                                        "range": [20, 40],
                                        "color": "#8CB5FF",
                                    },

                                    {
                                        "range": [40, 60],
                                        "color": "#CFCFCF",
                                    },

                                    {
                                        "range": [60, 80],
                                        "color": "#F4A6A6",
                                    },

                                    {
                                        "range": [80, 100],
                                        "color": "#D62828",
                                    },
                                ],

                                "threshold": {
                                    "line": {
                                        "color": "black",
                                        "width": 5,
                                    },
                                    "thickness": 0.85,
                                    "value": orientation_score,
                                },
                            },
                        )
                    )

                    # Small downward arrow over the threshold line
                    x_pos = 0.18 + (orientation_score / 100) * 0.64


                    fig.update_layout(
                        height=250,
                        margin=dict(
                            l=10,
                            r=10,
                            t=20,
                            b=10,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                else:

                    st.info("No political orientation detected.")

                # --------------------------------------------------------

                # Political Salience Badge

                if salience_row is not None:

                    salience_value = salience_row["salience"]

                    salience_colors = {
                        "low": "#2e7d32",
                        "medium": "#f9a825",
                        "high": "#c62828",
                    }

                    color = salience_colors.get(
                        salience_value.lower(),
                        "#757575",
                    )

                    st.markdown(
                        f"""
                        <div style="text-align:center;margin-top:-8px;">
                            <span style="
                                background:{color}22;
                                color:{color};
                                padding:4px 12px;
                                border-radius:999px;
                                font-size:.85rem;
                                font-weight:600;
                                border:1px solid {color};
                            ">
                                Political Salience: {salience_value}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # # ============================================================
            # # Article Text with Sentence Sentiment
            # # ============================================================

            # st.markdown(
            #     "### Article Text"
            # )


            # highlighted_text = sentiment_highlight_text(
            #     sentence_sent
            # )


            # if highlighted_text:

            #     article_html = f"""
            #     <div style="
            #         font-family: Inter, Arial, sans-serif;
            #         font-size:1.05rem;
            #         line-height:1.9;
            #         padding:20px;
            #         border:1px solid #dddddd;
            #         border-radius:10px;
            #         color:#222;
            #     ">
            #         {highlighted_text}
            #     </div>
            #     """


            #     st.components.v1.html(
            #         article_html,
            #         height=600,
            #         scrolling=True
            #     )


            # else:

            #     st.info(
            #         "No sentence-level sentiment available."
            #     )

            # ============================================================
            # Secondary Tab
            # ============================================================


            st.divider()


            with st.expander(
                "Analysis Details",
                expanded=False
            ):
                st.subheader(
                    "Classification Details"
                )


                analysis_cols = st.columns(
                    2
                )


                with analysis_cols[0]:


                    st.markdown(
                        "### Topics"
                    )


                    if not topics.empty:

                        st.dataframe(
                            topics,
                            hide_index=True,
                            use_container_width=True
                        )

                    else:

                        st.write(
                            "No topic classifications."
                        )



                    st.markdown(
                        "### Article Format"
                    )


                    if not formats.empty:

                        st.dataframe(
                            formats,
                            hide_index=True,
                            use_container_width=True
                        )

                    else:

                        st.write(
                            "No format classification."
                        )



                with analysis_cols[1]:


                    st.markdown(
                        "### SIC Classification"
                    )


                    if not sic.empty:

                        st.dataframe(
                            sic,
                            hide_index=True,
                            use_container_width=True
                        )

                    else:

                        st.write(
                            "No SIC classification."
                        )


                    st.markdown(
                        "### Political Analysis"
                    )


                    if not orientation.empty:

                        st.dataframe(
                            orientation,
                            hide_index=True,
                            use_container_width=True
                        )


                    if not salience.empty:

                        st.dataframe(
                            salience,
                            hide_index=True,
                            use_container_width=True
                        )

            # ============================================================
            # Footer Information
            # ============================================================


            st.divider()


            footer_cols = st.columns(
                3
            )


            with footer_cols[0]:

                st.caption(
                    f"""
                    Article ID:
                    {article['article_id']}
                    """
                )


            with footer_cols[1]:

                st.caption(
                    f"""
                    Processed source:
                    {article['hostname']
                    or 'Unknown'}
                    """
                )


            with footer_cols[2]:

                if article["url"]:

                    st.caption(
                        "Original URL available above"
                    )


            # ============================================================
            # End of article_detail.py
            # ============================================================

    with reader_col:

            col1,col2 = st.columns([5,1])

            with col1:
                st.subheader("Highlighted Article")

            with col2:

                if st.button("✕"):

                    st.session_state.show_reader=False

                    st.rerun()

            st.markdown("### 📖 Highlighted Article")

            st.caption(article["title"])

            st.components.v1.html(
                article_html,
                height=950,
                scrolling=True
            )

else:

    dashboard_col = st.container()

    with dashboard_col: 
        hero_left, hero_right = st.columns(
            [1.15, 3],
            gap="large"
        )


        with hero_left:

            if article["image_url"]:

                st.markdown(
                    "<div class='hero-img'>",
                    unsafe_allow_html=True
                )

                st.image(
                    article["image_url"],
                    width=260
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            else:

                st.info("No article image available")


        with hero_right:

            st.markdown(
                f"""
                <div style="
                    font-size:1.55rem;
                    font-weight:700;
                    line-height:1.2;
                    margin-bottom:.25rem;
                ">
                {article['title']}
                </div>

                <div style="
                    font-size:.85rem;
                    color:#888;
                ">
                {article['sitename'] or article['hostname']}
                &nbsp; • &nbsp;
                {parse_date(article['publish_date'])}
                </div>
                """,
                unsafe_allow_html=True
                )


            if article["description"]:

                st.markdown(
                f"""
                <div style="
                font-size:.9rem;
                color:#555;
                max-height:55px;
                overflow:hidden;
                ">
                {article['description']}
                </div>
                """,
                unsafe_allow_html=True
                )


            meta_cols = st.columns(4)


            with meta_cols[0]:

                if article["author"]:

                    st.markdown(
                        f"""
                        **Author**

                        {article['author']}
                        """
                    )


            with meta_cols[1]:

                st.markdown(
                    f"""
                    **Language**

                    {article['language'] or "Unknown"}
                    """
                )


            with meta_cols[2]:

                if article["url"]:

                    st.link_button(
                        "Open Original Article",
                        article["url"]
                    )

            with meta_cols[3]:
                if st.button("📖 Read Article"):
                    st.session_state.show_reader = not st.session_state.show_reader



        # ============================================================
        # Metadata Strip
        # ============================================================

        topic_summary = "N/A"


        if topics is not None and not topics.empty:

            topic_rows = topics[
                topics["confidence"] >= CONFIDENCE_THRESHOLDS["topic"]
            ].sort_values(
                "confidence",
                ascending=False
            )


            if not topic_rows.empty:

                topic_summary = " • ".join(
                    topic_rows["topic"].tolist()
                )

        format_summary = "N/A"


        if formats is not None and not formats.empty:

            format_rows = formats.sort_values(
                "confidence",
                ascending=False
            )


            format_summary = format_rows.iloc[0]["format"]

        st.divider()


        meta = [
            ("Words", f"{article['word_count']:,}"),
            ("Source", article["sitename"]),
            # ("Language", article["language"]),
            ("Published", parse_date(article["publish_date"])),
            ("Modified", parse_date(article["modified_date"])),
            ("Topics", topic_summary),
            ("Format", format_summary)
        ]


        cols = st.columns(6)


        for col, (label,value) in zip(cols,meta):

            with col:

                st.markdown(
                f"""
                <div style="
                    border:1px solid rgba(120,120,120,.2);
                    border-radius:8px;
                    padding:8px;
                    height:55px;
                ">

                <div style="
                    font-size:.65rem;
                    color:#888;
                    text-transform:uppercase;
                ">
                {label}
                </div>

                <div style="
                    font-size:.95rem;
                    font-weight:600;
                ">
                {value}
                </div>

                </div>
                """,
                unsafe_allow_html=True
                )

        # ============================================================
        # Intelligence Summary
        # ============================================================


        st.divider()

        # st.markdown(
        #     """
        #     <div class='section-title'>
        #     Intelligence Summary
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )



        def render_intel_card(
                title,
                value,
                color="#888"
        ):

            st.markdown(
                f"""
                <div style="
                    border:1px solid rgba(120,120,120,.25);
                    border-left:5px solid {color};
                    border-radius:12px;
                    padding:12px;
                    min-height:100px;
                ">

                <div style="
                    font-size:.7rem;
                    color:#888;
                    text-transform:uppercase;
                ">
                {title}
                </div>


                <div style="
                    font-size:1rem;
                    font-weight:700;
                    margin-top:8px;
                    color:{color};
                ">
                {value}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )



        # ============================================================
        # Build Intelligence Objects
        # ============================================================

        intel_cards = []


        # # Topics

        # if topics is not None and not topics.empty:

        #     topic_rows = (
        #         topics[
        #             topics["confidence"] >= CONFIDENCE_THRESHOLDS["topic"]
        #         ]
        #         .sort_values(
        #             "confidence",
        #             ascending=False
        #         )
        #     )


        #     if not topic_rows.empty:

        #         intel_cards.append(
        #             (
        #                 "Topics",
        #                 " • ".join(
        #                     topic_rows["topic"].tolist()
        #                 ),
        #                 "#2d6cdf"
        #             )
        #         )



        # # Article Format

        # fmt = best_prediction(
        #     formats,
        #     "format",
        #     CONFIDENCE_THRESHOLDS["format"]
        # )

        # if fmt is not None:

        #     intel_cards.append(
        #         (
        #             "Format",
        #             fmt["format"],
        #             "#6f42c1"
        #         )
        #     )


        # Sentiment

        # if sentiment is not None:

        #     score = sentiment["sentiment_score"]

        #     label = (
        #         "Positive"
        #         if score > .15
        #         else
        #         "Negative"
        #         if score < -.15
        #         else
        #         "Neutral"
        #     )

        #     intel_cards.append(
        #         (
        #             "Overall Sentiment",
        #             label,
        #             None,
        #             sentiment_color(score)
        #         )
        #     )



        # ============================================================
        # Render Cards
        # ============================================================

        # if intel_cards:


        #     card_cols = st.columns(
        #         4
        #     )


        #     for index, card in enumerate(
        #         intel_cards
        #     ):

        #         with card_cols[
        #             index % 4
        #         ]:

        #             render_intel_card(
        #                 card[0],
        #                 card[1],
        #                 card[2],
        #             )

        # else:

        #     st.info(
        #         "No classifications exceeded the confidence threshold."
        #     )


        # ============================================================
        # Sentiment Summary Panel
        # ============================================================

        if sentiment is not None:

            st.markdown(
                """
                <div class='section-title'
                style='margin-top:1rem'>
                Article Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )

            if sentiment is not None:

                score = sentiment["sentiment_score"]


                if score > 0.15:
                    sentiment_label = "Positive"

                elif score < -0.15:
                    sentiment_label = "Negative"

                else:
                    sentiment_label = "Neutral"


                st.markdown(
                    f"""
                    <div style="
                        font-size:1.2rem;
                        font-weight:700;
                        margin-bottom:15px;
                    ">
                    Overall Sentiment:
                    <span style="
                        color:{sentiment_color(score)};
                    ">
                    {sentiment_label}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            sent_cols = st.columns(5)


            sentiment_metrics = [

                (
                    "Score",
                    round(
                        sentiment["sentiment_score"],
                        3
                    )
                ),

                (
                    "Positive",
                    int(sentiment["positive"])
                ),

                (
                    "Neutral",
                    int(sentiment["neutral"])
                ),

                (
                    "Negative",
                    int(sentiment["negative"])
                )

            ]


            for col, metric in zip(
                sent_cols,
                sentiment_metrics
            ):

                with col:

                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid rgba(120,120,120,.25);
                            border-radius:10px;
                            padding:10px;
                        ">

                        <div style="
                            color:#888;
                            font-size:.75rem;
                        ">
                        {metric[0]}
                        </div>

                        <div style="
                            font-size:1.2rem;
                            font-weight:700;
                        ">
                        {metric[1]}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ============================================================
        # Entity Sentiment Treemap
        # ============================================================


        st.markdown(
            """
            <div class='section-title'>
            Entity Intelligence Map
            </div>
            """,
            unsafe_allow_html=True
        )


        if (
            entity_sent is not None
            and not entity_sent.empty
        ):


            entity_map = entity_sent.copy()


            # Add mention counts

            if (
                entities is not None
                and not entities.empty
            ):

                entity_map = entity_map.merge(
                    entities[
                        [
                            "entity_text",
                            "entity_label",
                            "mention_count"
                        ]
                    ],
                    on="entity_text",
                    how="left"
                )


            else:

                entity_map["mention_count"] = 1

                entity_map["entity_label"] = "Other"



            # Convert sentiment numeric

            entity_map["sentiment"] = pd.to_numeric(
                entity_map["sentiment"],
                errors="coerce"
            )


            # Aggregate entities

            entity_summary = (

                entity_map
                .groupby(
                    [
                        "entity_text",
                        "entity_label"
                    ],
                    as_index=False
                )
                .agg(
                    {
                        "sentiment":"mean",
                        "mention_count":"max"
                    }
                )

            )


            # Remove missing values

            entity_summary = (
                entity_summary
                .dropna(
                    subset=[
                        "entity_text"
                    ]
                )
            )


            # Limit largest entities

            entity_summary = (
                entity_summary
                .sort_values(
                    "mention_count",
                    ascending=False
                )
                .head(40)
            )


            if not entity_summary.empty:


                fig = px.treemap(

                    entity_summary,

                    path=[
                        "entity_label",
                        "entity_text"
                    ],

                    values="mention_count",

                    color="sentiment",

                    color_continuous_scale=[
                        "red",
                        "white",
                        "green"
                    ],

                    range_color=[
                        -1,
                        1
                    ],

                )


                fig.update_traces(

                    texttemplate=
                    "<b>%{label}</b><br>%{value} mentions",

                    hovertemplate=
                    """
                    <b>%{label}</b><br>
                    Mentions: %{value}<br>
                    Sentiment: %{color:.2f}
                    <extra></extra>
                    """

                )


                fig.update_layout(

                    height=300,

                    margin=dict(
                        t=10,
                        l=5,
                        r=5,
                        b=5
                    ),

                    coloraxis_colorbar=dict(
                        title="Sentiment"
                    )

                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            else:

                st.info(
                    "No entity sentiment available."
                )


        else:

            st.info(
                "No entity sentiment data available."
            )

        # ============================================================
        # Industry + Political Intelligence
        # ============================================================

        industry_col, political_col = st.columns(
            [1,1],
            gap="large"
        )


        with industry_col:

            # ============================================================
            # SIC Hierarchy
            # ============================================================


            def get_sic(level):

                if sic is None or sic.empty:
                    return None

                row = (
                    sic[
                        sic["level"] == level
                    ]
                    .sort_values(
                        "confidence",
                        ascending=False
                    )
                )

                if row.empty:
                    return None

                return row.iloc[0]["name"]



            sic_division = get_sic("division")
            sic2 = get_sic("sic2")
            sic3 = get_sic("sic3")
            sic4 = get_sic("sic4")



            if any(
                [
                    sic_division,
                    sic2,
                    sic3,
                    sic4
                ]
            ):


                st.markdown(
                    """
                    <div class='section-title'>
                    Industry Classification
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(

                    f"""
                    <div style="
                        border-radius:14px;
                        border:1px solid rgba(120,120,120,.25);
                        padding:16px;
                        background:rgba(0,0,0,.02);
                    ">


                    <div style="
                        font-size:.75rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    SIC Division
                    </div>


                    <div style="
                        font-size:1.35rem;
                        font-weight:800;
                        color:#1f77b4;
                        margin-bottom:14px;
                    ">
                    {sic_division or "N/A"}
                    </div>



                    <div style="
                        font-size:.7rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    SIC 2
                    </div>


                    <div style="
                        font-size:1.1rem;
                        font-weight:700;
                        color:#2ca02c;
                        margin-bottom:12px;
                    ">
                    {sic2 or "N/A"}
                    </div>



                    <div style="
                        font-size:.65rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    SIC 3
                    </div>


                    <div style="
                        font-size:.95rem;
                        font-weight:600;
                        color:#9467bd;
                        margin-bottom:10px;
                    ">
                    {sic3 or "N/A"}
                    </div>



                    <div style="
                        font-size:.6rem;
                        color:#888;
                        text-transform:uppercase;
                    ">
                    SIC 4
                    </div>


                    <div style="
                        font-size:.85rem;
                        color:#d62728;
                        font-weight:500;
                    ">
                    {sic4 or "N/A"}
                    </div>


                    </div>
                    """,

                    unsafe_allow_html=True

                )


        # ============================================================
        # Political Intelligence
        # ============================================================

        with political_col:

            st.markdown("### Political Intelligence")

            orientation_row = best_prediction(
                orientation,
                "orientation",
                CONFIDENCE_THRESHOLDS["orientation"]
            )

            salience_row = best_prediction(
                salience,
                "salience",
                CONFIDENCE_THRESHOLDS["salience"]
            )

            if orientation_row is not None:

                orientation_value = orientation_row["orientation"]

                orientation_map = {
                    "Progressive or left-wing": 10,
                    "Center-left": 30,
                    "Centrist or politically neutral": 50,
                    "Center-right": 70,
                    "Right-wing or conservative": 90,
                }

                orientation_score = orientation_map.get(
                    orientation_value,
                    50
                )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge",

                        value=orientation_score,

                        gauge={

                            "shape": "angular",

                            "axis": {
                                "range": [0, 100],
                                "tickvals": [0, 25, 50, 75, 100],
                                "ticktext": [
                                    "Left",
                                    "Center<br>Left",
                                    "Neutral",
                                    "Center<br>Right",
                                    "Right",
                                ],
                            },

                            "bar": {
                                "color": "#333333",
                                "thickness": 0.20,
                            },

                            "steps": [

                                {
                                    "range": [0, 20],
                                    "color": "#2F6BFF",
                                },

                                {
                                    "range": [20, 40],
                                    "color": "#8CB5FF",
                                },

                                {
                                    "range": [40, 60],
                                    "color": "#CFCFCF",
                                },

                                {
                                    "range": [60, 80],
                                    "color": "#F4A6A6",
                                },

                                {
                                    "range": [80, 100],
                                    "color": "#D62828",
                                },
                            ],

                            "threshold": {
                                "line": {
                                    "color": "black",
                                    "width": 5,
                                },
                                "thickness": 0.85,
                                "value": orientation_score,
                            },
                        },
                    )
                )

                # Small downward arrow over the threshold line
                x_pos = 0.18 + (orientation_score / 100) * 0.64


                fig.update_layout(
                    height=250,
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            else:

                st.info("No political orientation detected.")

            # --------------------------------------------------------

            # Political Salience Badge

            if salience_row is not None:

                salience_value = salience_row["salience"]

                salience_colors = {
                    "low": "#2e7d32",
                    "medium": "#f9a825",
                    "high": "#c62828",
                }

                color = salience_colors.get(
                    salience_value.lower(),
                    "#757575",
                )

                st.markdown(
                    f"""
                    <div style="text-align:center;margin-top:-8px;">
                        <span style="
                            background:{color}22;
                            color:{color};
                            padding:4px 12px;
                            border-radius:999px;
                            font-size:.85rem;
                            font-weight:600;
                            border:1px solid {color};
                        ">
                            Political Salience: {salience_value}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # # ============================================================
        # # Article Text with Sentence Sentiment
        # # ============================================================

        # st.markdown(
        #     "### Article Text"
        # )


        # highlighted_text = sentiment_highlight_text(
        #     sentence_sent
        # )


        # if highlighted_text:

        #     article_html = f"""
        #     <div style="
        #         font-family: Inter, Arial, sans-serif;
        #         font-size:1.05rem;
        #         line-height:1.9;
        #         padding:20px;
        #         border:1px solid #dddddd;
        #         border-radius:10px;
        #         color:#222;
        #     ">
        #         {highlighted_text}
        #     </div>
        #     """


        #     st.components.v1.html(
        #         article_html,
        #         height=600,
        #         scrolling=True
        #     )


        # else:

        #     st.info(
        #         "No sentence-level sentiment available."
        #     )

        # ============================================================
        # Secondary Tab
        # ============================================================


        st.divider()


        with st.expander(
            "Analysis Details",
            expanded=False
        ):
            st.subheader(
                "Classification Details"
            )


            analysis_cols = st.columns(
                2
            )


            with analysis_cols[0]:


                st.markdown(
                    "### Topics"
                )


                if not topics.empty:

                    st.dataframe(
                        topics,
                        hide_index=True,
                        use_container_width=True
                    )

                else:

                    st.write(
                        "No topic classifications."
                    )



                st.markdown(
                    "### Article Format"
                )


                if not formats.empty:

                    st.dataframe(
                        formats,
                        hide_index=True,
                        use_container_width=True
                    )

                else:

                    st.write(
                        "No format classification."
                    )



            with analysis_cols[1]:


                st.markdown(
                    "### SIC Classification"
                )


                if not sic.empty:

                    st.dataframe(
                        sic,
                        hide_index=True,
                        use_container_width=True
                    )

                else:

                    st.write(
                        "No SIC classification."
                    )


                st.markdown(
                    "### Political Analysis"
                )


                if not orientation.empty:

                    st.dataframe(
                        orientation,
                        hide_index=True,
                        use_container_width=True
                    )


                if not salience.empty:

                    st.dataframe(
                        salience,
                        hide_index=True,
                        use_container_width=True
                    )

        # ============================================================
        # Footer Information
        # ============================================================


        st.divider()


        footer_cols = st.columns(
            3
        )


        with footer_cols[0]:

            st.caption(
                f"""
                Article ID:
                {article['article_id']}
                """
            )


        with footer_cols[1]:

            st.caption(
                f"""
                Processed source:
                {article['hostname']
                or 'Unknown'}
                """
            )


        with footer_cols[2]:

            if article["url"]:

                st.caption(
                    "Original URL available above"
                )


        # ============================================================
        # End of article_detail.py
        # ============================================================
