# ============================================================
# article_detail.py  — v2
# ============================================================

from __future__ import annotations

import re
import math
import html
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

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
    "orientation": 0.15,
    "salience": 0.0,
    "intent": 0.30,
}

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 0.75rem;
    max-width: 1700px;
}
.hero-img img { border-radius: 10px; }
[data-testid="stAppViewContainer"] { overflow-y: auto; }
[data-testid="stHeader"] { z-index: 999; }
/* Compact vertical rhythm */
[data-testid="stVerticalBlock"] { gap: 0.7rem; }
[data-testid="stImage"] { margin-bottom: 0; }
hr { margin: 0.6rem 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Utility helpers
# ============================================================


def sentiment_color(score: float) -> str:
    if score >= 0.1:
        return "#1b9e4b"
    if score <= -0.1:
        return "#d93025"
    return "#888888"


def orientation_color(label: str) -> str:
    label = str(label).lower()
    if "left" in label:
        return "#2d6cdf"
    if "right" in label:
        return "#d93025"
    return "#7a7a7a"


def parse_date(value) -> str:
    if not value:
        return ""
    try:
        return pd.to_datetime(value).strftime("%b %d, %Y")
    except Exception:
        return str(value)


def sentiment_highlight_text(df) -> str | None:
    if df is None or df.empty:
        return None
    parts = []
    for _, row in df.iterrows():
        sentence = html.escape(str(row["sentence"]))
        sentiment = str(row["sentiment"]).lower()
        score = float(row["score"])
        if sentiment == "positive":
            bg = f"rgba(46,204,113,{(abs(score) ** 0.7) * 0.35})"
        elif sentiment == "negative":
            bg = f"rgba(231,76,60,{(abs(score) ** 0.7) * 0.35})"
        else:
            bg = "transparent"
        parts.append(
            f'<span style="background-color:{bg};padding:2px 4px;border-radius:4px;" '
            f'title="Sentiment: {sentiment} | Score: {score:.2f}">{sentence}</span>'
        )
    return " ".join(parts)


# ============================================================
# Readability
# ============================================================


def _count_syllables(word: str) -> int:
    vowels = re.findall(r"[aeiou]+", word.lower())
    count = len(vowels)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_readability(text: str) -> dict:
    if not text:
        return {"grade_level": 8.0, "reading_ease": 50.0, "vocab_diversity": 0.5}

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    n_words = max(len(words), 1)
    n_syllables = sum(_count_syllables(w) for w in words)
    unique_words = len(set(w.lower() for w in words))

    grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    ease = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    return {
        "grade_level": max(0.0, grade),
        "reading_ease": max(0.0, min(100.0, ease)),
        "vocab_diversity": unique_words / n_words,
    }


def audience_score_and_label(grade_level: float) -> tuple[float, str]:
    score = max(0.0, min(100.0, (grade_level - 6) / 12 * 100))
    if score < 30:
        label = "General Public"
    elif score < 60:
        label = "Informed Reader"
    elif score < 85:
        label = "Professional"
    else:
        label = "Expert / Specialist"
    return score, label


def maturity_score_and_label(reading_ease: float, vocab_diversity: float) -> tuple[float, str]:
    complexity = (100 - reading_ease) / 100
    score = max(0.0, min(100.0, (complexity * 0.6 + vocab_diversity * 0.4) * 100))
    if score < 25:
        label = "Accessible"
    elif score < 50:
        label = "Standard"
    elif score < 75:
        label = "Sophisticated"
    else:
        label = "Advanced"
    return score, label


def spectrum_bar_html(title: str, score: float, left_label: str, right_label: str, current_label: str) -> str:
    pct = max(2, min(98, score))
    return f"""
<div style="padding:7px 12px;border:1px solid rgba(120,120,120,.25);
            border-radius:10px;margin-bottom:6px;">
  <div style="font-size:.6rem;color:#888;text-transform:uppercase;
              letter-spacing:.05rem;margin-bottom:5px;">{title}</div>
  <div style="position:relative;height:7px;border-radius:4px;
              background:linear-gradient(to right,#4a9eff,#9c27b0);margin-bottom:4px;">
    <div style="position:absolute;left:{pct}%;top:-5px;transform:translateX(-50%);
                width:18px;height:18px;border-radius:50%;
                background:white;border:2px solid #555;
                box-shadow:0 1px 4px rgba(0,0,0,.4);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:.65rem;color:#888;">
    <span>{left_label}</span>
    <span style="font-weight:700;color:#ddd;">{current_label}</span>
    <span>{right_label}</span>
  </div>
</div>
"""


def small_card_html(
    label: str,
    value: str,
    color: str = "#888",
    text_color: str = "#000000",
    subtitle: str = "",
) -> str:
    subtitle_html = (
        f'<div style="font-size:.62rem;color:#888;margin-top:1px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{subtitle}</div>'
        if subtitle
        else ""
    )
    return f"""
<div style="border:1px solid rgba(120,120,120,.2);border-left:4px solid {color};
            border-radius:8px;padding:8px 10px;min-height:52px;overflow:hidden;">
  <div style="font-size:.60rem;color:#888;text-transform:uppercase;letter-spacing:.04rem;">{label}</div>
  <div style="font-size:.88rem;font-weight:600;margin-top:2px;color:{text_color};
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>
  {subtitle_html}
</div>
"""


# ============================================================
# Word Cloud
# ============================================================


def build_entity_wordcloud(entity_sent_df, entities_df):
    """Returns a PIL Image of the entity word cloud, or None."""
    if entity_sent_df is None or entity_sent_df.empty:
        return None

    merged = entity_sent_df.copy()
    if entities_df is not None and not entities_df.empty:
        merged = merged.merge(
            entities_df[["entity_text", "mention_count"]],
            on="entity_text",
            how="left",
        )
    else:
        merged["mention_count"] = 1

    merged["sentiment"] = pd.to_numeric(merged["sentiment"], errors="coerce").fillna(0)
    merged["mention_count"] = merged["mention_count"].fillna(1).astype(int)
    merged = merged.dropna(subset=["entity_text"])
    merged = (
        merged.groupby("entity_text", as_index=False)
        .agg({"sentiment": "mean", "mention_count": "max"})
    )

    if merged.empty:
        return None

    freq = dict(zip(merged["entity_text"], merged["mention_count"]))
    sentiment_map = dict(zip(merged["entity_text"], merged["sentiment"]))

    # Normalise against the strongest sentiment in this article so the full
    # light→dark range is used (floor avoids over-amplifying tiny scores).
    max_abs = max(0.3, float(merged["sentiment"].abs().max()))

    def _blend(pale, dark, t):
        return tuple(int(round(p + (d - p) * t)) for p, d in zip(pale, dark))

    def color_func(word, font_size=None, position=None, orientation=None, random_state=None, **kwargs):
        score = sentiment_map.get(word, 0.0)
        if -0.1 < score < 0.1:
            return "rgb(150,150,150)"
        t = min(1.0, abs(score) / max_abs)  # 0 = neutral tint, 1 = deepest
        if score > 0:
            r, g, b = _blend((150, 205, 165), (12, 105, 45), t)  # pale → deep green
        else:
            r, g, b = _blend((225, 165, 160), (150, 25, 20), t)  # pale → deep red
        return f"rgb({r},{g},{b})"

    wc = WordCloud(
        width=900,
        height=230,
        background_color=None,
        mode="RGBA",
        color_func=color_func,
        prefer_horizontal=0.8,
        max_words=40,
        collocations=False,
    ).generate_from_frequencies(freq)

    return wc.to_image()


# ============================================================
# Analysis Charts
# ============================================================


def section_label_html(text: str) -> str:
    return (
        '<div style="font-size:.62rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:.05rem;margin-bottom:6px;">{text}</div>'
    )


def sic_hierarchy_html(sic_df) -> str | None:
    """Vertical list of every SIC hierarchy level: division → sic2 → sic3 → sic4."""

    def get_level(level):
        if sic_df is None or sic_df.empty:
            return None
        rows = sic_df[sic_df["level"] == level].sort_values("confidence", ascending=False)
        return None if rows.empty else rows.iloc[0]["name"]

    levels = [
        ("SIC Division", get_level("division"), "1.15rem", "800", "#1f77b4"),
        ("SIC 2", get_level("sic2"), "1.00rem", "700", "#2ca02c"),
        ("SIC 3", get_level("sic3"), "0.90rem", "600", "#9467bd"),
        ("SIC 4", get_level("sic4"), "0.82rem", "500", "#d62728"),
    ]

    if not any(value for _, value, *_ in levels):
        return None

    rows_html = ""
    for label, value, size, weight, color in levels:
        rows_html += (
            f'<div style="font-size:.58rem;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04rem;">{label}</div>'
            f'<div style="font-size:{size};font-weight:{weight};color:{color};'
            f'line-height:1.15;margin-bottom:7px;">{value or "N/A"}</div>'
        )

    return (
        '<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);'
        f'padding:10px 14px;background:rgba(0,0,0,.02);">{rows_html}</div>'
    )


def build_orientation_gauge(orientation_row):
    """Angular gauge placing political orientation on a Left→Right scale."""
    if orientation_row is None:
        return None

    orientation_map = {
        "Progressive or left-wing": 10,
        "Center-left": 30,
        "Centrist or politically neutral": 50,
        "Center-right": 70,
        "Right-wing or conservative": 90,
    }
    score = orientation_map.get(orientation_row["orientation"], 50)

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["Left", "Center<br>Left", "Neutral", "Center<br>Right", "Right"],
                    "tickcolor": "#888",
                    "tickfont": {"color": "#888", "size": 10},
                },
                # Hide the default value bar — the needle below shows the level.
                "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
                "steps": [
                    {"range": [0, 20], "color": "#2F6BFF"},
                    {"range": [20, 40], "color": "#8CB5FF"},
                    {"range": [40, 60], "color": "#CFCFCF"},
                    {"range": [60, 80], "color": "#F4A6A6"},
                    {"range": [80, 100], "color": "#D62828"},
                ],
            },
        )
    )

    # Needle: an arrow from the gauge centre pointing at the score, drawn with
    # paper-coordinate shapes (annotation arrows don't allow axref="paper").
    # The half-gauge sweeps 180° (score 0 = left) to 0° (score 100 = right).
    theta = math.pi * (1 - score / 100.0)
    cx, cy = 0.5, 0.11          # gauge centre in paper coords
    rx, ry = 0.40, 0.42         # needle reach (rx/ry tune for aspect ratio)
    tip_x = cx + rx * math.cos(theta)
    tip_y = cy + ry * math.sin(theta)

    # Unit direction + perpendicular (for the arrowhead) in paper coords.
    dx, dy = tip_x - cx, tip_y - cy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len, head_w = 0.07, 0.035
    base_x, base_y = tip_x - ux * head_len, tip_y - uy * head_len

    # Needle shaft.
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=cx,
        y0=cy,
        x1=base_x,
        y1=base_y,
        line=dict(color="#222222", width=3),
    )
    # Arrowhead (filled triangle) at the tip.
    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=(
            f"M {tip_x},{tip_y} "
            f"L {base_x + px * head_w},{base_y + py * head_w} "
            f"L {base_x - px * head_w},{base_y - py * head_w} Z"
        ),
        fillcolor="#222222",
        line_color="#222222",
    )
    # Pivot dot at the centre.
    fig.add_shape(
        type="circle",
        xref="paper",
        yref="paper",
        x0=cx - 0.018,
        y0=cy - 0.018,
        x1=cx + 0.018,
        y1=cy + 0.018,
        fillcolor="#222222",
        line_color="#222222",
    )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=18, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
    )
    return fig


def salience_badge_html(salience_value) -> str:
    """Pill badge summarising political salience level."""
    salience_colors = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}
    color = salience_colors.get(str(salience_value).lower(), "#757575")
    return f"""
    <div style="text-align:center;margin-top:-8px;">
        <span style="background:{color}22;color:{color};padding:4px 12px;
            border-radius:999px;font-size:.85rem;font-weight:600;border:1px solid {color};">
            Political Salience: {salience_value}
        </span>
    </div>
    """


# ============================================================
# Sports Highlights
# ============================================================


def is_sports_article(topics) -> bool:
    """True when the article's highest-confidence topic is Sports."""
    if topics is None or topics.empty:
        return False
    top = topics.sort_values("confidence", ascending=False).iloc[0]
    return str(top["topic"]).strip().lower() == "sports"


def _allowed_entities(roles, role_name: str):
    """
    Set of entity_texts the NLP role-classifier assigned to `role_name`.

    Returns None when no classification exists for this article yet, which
    signals callers to fall back to showing every entity of the NER label
    (unfiltered) rather than hiding everything.
    """
    if roles is None or roles.empty:
        return None
    return set(roles[roles["role"] == role_name]["entity_text"])


def _entity_chips(entities, label: str, accent: str, allowed=None, roles=None, limit: int = 10) -> str | None:
    """
    Pill list of the top entities of a given NER label, or None if none.

    `allowed` (a set of entity_texts) restricts the list to entities the NLP
    classifier confirmed for this role; None means no filtering.
    """
    if entities is None or entities.empty:
        return None
    df = entities[entities["entity_label"] == label]
    if allowed is not None:
        df = df[df["entity_text"].isin(allowed)]
    if df.empty:
        return None
    df = df.sort_values("mention_count", ascending=False).head(limit)

    entity_urls = roles.set_index("entity_text")["url"].to_dict()

    chips = ""
    for _, row in df.iterrows():
        name = html.escape(str(row["entity_text"]))
        # count = int(row["mention_count"]) if pd.notna(row["mention_count"]) else 1
        # NOT SHOWING COUNT FOR NOW
        count = 1
        badge = (
            f'<span style="opacity:.55;font-size:.62rem;margin-left:5px;">{count}</span>'
            if count > 1
            else ""
        )
        url = (entity_urls.get(name))
        chips += (
            f'<a href="{url}" target="_blank" class="entity-link">'
            f'<span style="display:inline-block;background:{accent}18;color:{accent};'
            f'border:1px solid {accent}55;border-radius:999px;padding:3px 10px;'
            f'margin:0 6px 7px 0;font-size:.78rem;font-weight:600;">{name}{badge}</span></a>'
        )
    return chips


def sports_highlights_html(entities, roles) -> str | None:
    """Card listing the key athletes and major events named in the article."""
    athletes = _entity_chips(entities, "PERSON", "#1f77b4", allowed=_allowed_entities(roles, 'athlete'), roles=roles)
    events = _entity_chips(entities, "EVENT", "#d62728", allowed=_allowed_entities(roles,'sporting_event'), roles=roles,)
    if athletes is None and events is None:
        return None

    def _section(sub_label: str, chips: str | None) -> str:
        if not chips:
            return ""
        return (
            f'<div style="font-size:.58rem;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04rem;margin:2px 0 7px;">{sub_label}</div>'
            f'<div style="margin-bottom:10px;">{chips}</div>'
        )

    body = _section("Athletes - Top 10", athletes) + _section("Major Events", events)
    return (
        '<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);'
        f'padding:12px 14px;background:rgba(0,0,0,.02);">{body}</div>'
    )


# ============================================================
# Database Loaders
# ============================================================


def load_articles():
    return query("SELECT * FROM articles ORDER BY publish_date DESC")


def load_article(article_id):
    df = query("SELECT * FROM articles WHERE article_id = ?", (article_id,))
    return None if df.empty else df.iloc[0]


def load_topics(article_id):
    return query("SELECT * FROM article_topics WHERE article_id = ? ORDER BY rank", (article_id,))


def load_intents(article_id):
    return query("SELECT * FROM article_intents WHERE article_id = ? ORDER BY rank", (article_id,))


def load_formats(article_id):
    return query("SELECT * FROM article_format WHERE article_id = ? ORDER BY rank", (article_id,))


def load_sic(article_id):
    return query("SELECT * FROM article_sic WHERE article_id = ?", (article_id,))


def load_sentiment(article_id):
    df = query("SELECT * FROM article_sentiment WHERE article_id = ?", (article_id,))
    return None if df.empty else df.iloc[0]


def load_sentence_sentiment(article_id):
    return query(
        "SELECT sentence, sentiment, score FROM sent_sentiment WHERE article_id = ? ORDER BY id",
        (article_id,),
    )


def load_orientation(article_id):
    return query(
        "SELECT * FROM political_orientation WHERE article_id = ? ORDER BY rank", (article_id,)
    )


def load_salience(article_id):
    return query(
        "SELECT * FROM political_salience WHERE article_id = ? ORDER BY rank", (article_id,)
    )


def load_entities(article_id):
    return query(
        "SELECT * FROM entities WHERE article_id = ? ORDER BY mention_count DESC", (article_id,)
    )


def load_entity_sentiment(article_id):
    return query("SELECT * FROM entity_sentiment WHERE article_id = ?", (article_id,))

def load_roles(article_id):
    return query("SELECT * FROM entity_roles WHERE article_id = ?", (article_id,))


def load_readability(article_id):
    """NLP-classified audience/maturity, or None if not yet classified."""
    try:
        df = query("SELECT * FROM article_readability WHERE article_id = ?", (article_id,))
    except Exception:
        return None  # table not created yet
    return None if df.empty else df.iloc[0]


# ============================================================
# Selection helpers
# ============================================================


def best_prediction(df, value_col, threshold):
    if df is None or df.empty:
        return None
    filtered = df[df["confidence"] >= threshold]
    if filtered.empty:
        return None
    return filtered.sort_values("confidence", ascending=False).iloc[0]


def best_sic(df, level):
    if df is None or df.empty:
        return None
    tmp = df[df["level"] == level]
    if tmp.empty:
        return None
    row = tmp.sort_values("confidence", ascending=False).iloc[0]
    return row if row["confidence"] >= CONFIDENCE_THRESHOLDS["sic"] else None


# ============================================================
# Article Selector
# ============================================================

articles = load_articles()

if articles.empty:
    st.error("No articles found.")
    st.stop()

article_lookup = {
    f"{row['title'][:60]} | {row['sitename']} | ID:{row['article_id']}": row["article_id"]
    for _, row in articles.iterrows()
}

st.sidebar.write("Articles loaded:", len(articles))

selection = st.selectbox(
    "Choose Article",
    list(article_lookup.keys()),
    label_visibility="collapsed",
)
ARTICLE_ID = article_lookup[selection]

# ============================================================
# Load Data
# ============================================================

article = load_article(ARTICLE_ID)
topics = load_topics(ARTICLE_ID)
intents = load_intents(ARTICLE_ID)
formats = load_formats(ARTICLE_ID)
sic = load_sic(ARTICLE_ID)
sentiment = load_sentiment(ARTICLE_ID)
orientation = load_orientation(ARTICLE_ID)
salience = load_salience(ARTICLE_ID)
entities = load_entities(ARTICLE_ID)
entity_sent = load_entity_sentiment(ARTICLE_ID)
sentence_sent = load_sentence_sentiment(ARTICLE_ID)
readability_row = load_readability(ARTICLE_ID)
roles = load_roles(ARTICLE_ID)

highlighted_text = sentiment_highlight_text(sentence_sent)

article_html = (
    f'<div style="font-family:Georgia,serif;font-size:1.05rem;line-height:1.9;padding:20px;">'
    f"{highlighted_text or ''}</div>"
)

# Audience / writing maturity: prefer NLP classification, fall back to the
# Flesch-based heuristics when the article hasn't been classified yet.
readability = compute_readability(article["text"] or "")  # kept for Analysis Details
if readability_row is not None:
    aud_score = float(readability_row["audience_score"])
    aud_label = str(readability_row["audience_label"])
    mat_score = float(readability_row["maturity_score"])
    mat_label = str(readability_row["maturity_label"])
    readability_source = "NLP (zero-shot)"
else:
    aud_score, aud_label = audience_score_and_label(readability["grade_level"])
    mat_score, mat_label = maturity_score_and_label(
        readability["reading_ease"], readability["vocab_diversity"]
    )
    readability_source = "Heuristic (Flesch-Kincaid)"

# ============================================================
# Layout
# ============================================================

if st.session_state.show_reader:
    dashboard_col, reader_col = st.columns([3, 2], gap="large")
else:
    dashboard_col = st.container()
    reader_col = None


def render_dashboard(col):
    with col:

        # ====================================================
        # ROW 1: Hero
        # ====================================================

        img_col, text_col = st.columns([1, 3], gap="large")

        with img_col:
            if article["image_url"]:
                st.image(article["image_url"], use_container_width=True)
            else:
                st.markdown(
                    '<div style="background:rgba(120,120,120,.1);border-radius:10px;height:120px;'
                    'display:flex;align-items:center;justify-content:center;color:#666;font-size:.8rem;">'
                    "No image</div>",
                    unsafe_allow_html=True,
                )

        with text_col:
            wc = article["word_count"] or 0
            read_time = max(1, round(wc / 200))
            meta_bits = [
                article["sitename"] or article["hostname"] or "Unknown source",
                parse_date(article["publish_date"]) or "N/A",
                f"By {article['author']}" if article["author"] else "Unknown author",
                f"{wc:,} words",
                f"~{read_time} min read",
            ]
            meta_line = " &nbsp;•&nbsp; ".join(str(b) for b in meta_bits)
            st.markdown(
                f"""
                <div style="font-size:1.45rem;font-weight:700;line-height:1.2;margin-bottom:.2rem;">
                    {article["title"]}
                </div>
                <div style="font-size:.78rem;color:#888;">
                    {meta_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if article["description"]:
                st.markdown(
                    f'<div style="font-size:.85rem;color:#666;max-height:42px;overflow:hidden;margin-top:4px;">'
                    f'{article["description"]}</div>',
                    unsafe_allow_html=True,
                )
            btn_a, btn_b, _ = st.columns([2, 2, 5])
            with btn_a:
                if article["url"]:
                    st.link_button("Open Article", article["url"], use_container_width=True)
            with btn_b:
                if st.button("Read", use_container_width=True):
                    st.session_state.show_reader = not st.session_state.show_reader
                    st.rerun()

        st.divider()

        # ====================================================
        # ROW 2: Inferred Intelligence
        # ====================================================

        # Sentiment
        sent_score = sentiment["sentiment_score"] if sentiment is not None else 0.0
        if sent_score > 0.15:
            sent_val = "Positive"
        elif sent_score < -0.15:
            sent_val = "Negative"
        else:
            sent_val = "Neutral"
        sent_color = sentiment_color(sent_score) if sentiment is not None else "#888"
        # Sentiment value text: green (positive) / red (negative) / black (neutral)
        if sent_score > 0.15:
            sent_text_color = "#1b9e4b"
        elif sent_score < -0.15:
            sent_text_color = "#d93025"
        else:
            sent_text_color = "#000000"

        # Topics — show confident topics, else fall back to the top topic
        # flagged as low-confidence rather than blanking to N/A.
        topic_val = "N/A"
        topic_subtitle = ""
        if topics is not None and not topics.empty:
            ranked = topics.sort_values("confidence", ascending=False)
            above = ranked[ranked["confidence"] >= CONFIDENCE_THRESHOLDS["topic"]]
            if not above.empty:
                topic_val = " • ".join(above["topic"].tolist()[:2])
            else:
                top = ranked.iloc[0]
                topic_val = str(top["topic"])
                topic_subtitle = f"Low confidence · {top['confidence']:.0%}"

        # Format
        fmt_row = best_prediction(formats, "format", CONFIDENCE_THRESHOLDS["format"])
        fmt_val = fmt_row["format"] if fmt_row is not None else "N/A"

        # Intent
        intent_row = best_prediction(intents, "intent", CONFIDENCE_THRESHOLDS["intent"])
        intent_val = intent_row["intent"] if intent_row is not None else "N/A"

        sent_subtitle = f"Score: {sent_score:+.2f}" if sentiment is not None else ""

        inferred_items = [
            ("Sentiment", sent_val, sent_color, sent_text_color, sent_subtitle),
            ("Topics", topic_val, "#2d6cdf", "#000000", topic_subtitle),
            ("Format", fmt_val, "#6f42c1", "#000000", ""),
            ("Intent", intent_val, "#fd7e14", "#000000", ""),
        ]

        for col, (label, value, color, text_color, subtitle) in zip(st.columns(4), inferred_items):
            with col:
                st.markdown(
                    small_card_html(label, value, color, text_color, subtitle),
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:2px;'></div>", unsafe_allow_html=True)

        # ====================================================
        # ROW 4: 2×2 Intelligence Grid
        #   ┌───────────────────┬───────────────────┐
        #   │ Entity Word Cloud │  Article Profile  │
        #   ├───────────────────┼───────────────────┤
        #   │ Industry (SIC)    │ Political Gauge   │
        #   └───────────────────┴───────────────────┘
        # ====================================================

        top_left, top_right = st.columns(2, gap="large")

        # --- Top-left: Entity Word Cloud ---
        with top_left:
            st.markdown(section_label_html("Entity Intelligence"), unsafe_allow_html=True)
            wc_image = build_entity_wordcloud(entity_sent, entities)
            if wc_image is not None:
                st.image(wc_image, use_container_width=True)
            else:
                st.info("No entity data available.")

        # --- Top-right: Audience & Writing Complexity spectrums ---
        with top_right:
            st.markdown(section_label_html("Article Profile"), unsafe_allow_html=True)
            st.markdown(
                spectrum_bar_html("Intended Audience", aud_score, "General", "Expert", aud_label),
                unsafe_allow_html=True,
            )
            st.markdown(
                spectrum_bar_html("Writing Complexity", mat_score, "Accessible", "Advanced", mat_label),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

        bottom_left, bottom_right = st.columns(2, gap="large")

        # --- Bottom-left: Hierarchical SIC industry list ---
        with bottom_left:
            st.markdown(section_label_html("Industry Classification"), unsafe_allow_html=True)
            sic_html = sic_hierarchy_html(sic)
            if sic_html is not None:
                st.markdown(sic_html, unsafe_allow_html=True)
            else:
                st.info("No industry classification available.")

        # --- Bottom-right: Sports highlights for sports articles, otherwise
        #     the political orientation gauge + salience badge. ---
        with bottom_right:
            if is_sports_article(topics):
                st.markdown(section_label_html("Sports Highlights"), unsafe_allow_html=True)
                sports_html = sports_highlights_html(entities, roles)
                if sports_html is not None:
                    st.markdown(sports_html, unsafe_allow_html=True)
                else:
                    st.info("No athletes or events detected.")
            else:
                st.markdown(section_label_html("Political Intelligence"), unsafe_allow_html=True)
                orient_row = best_prediction(orientation, "orientation", CONFIDENCE_THRESHOLDS["orientation"])
                gauge = build_orientation_gauge(orient_row)
                if gauge is not None:
                    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No political orientation detected.")

                salience_row = best_prediction(salience, "salience", CONFIDENCE_THRESHOLDS["salience"])
                if salience_row is not None:
                    st.markdown(salience_badge_html(salience_row["salience"]), unsafe_allow_html=True)

        # ====================================================
        # Analysis Details (expander)
        # ====================================================

        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

        with st.expander("Analysis Details", expanded=False):
            d_left, d_right = st.columns(2)
            with d_left:
                st.markdown("**Topics**")
                if topics is not None and not topics.empty:
                    st.dataframe(topics, hide_index=True, use_container_width=True)
                st.markdown("**Format**")
                if formats is not None and not formats.empty:
                    st.dataframe(formats, hide_index=True, use_container_width=True)
                st.markdown("**Intent**")
                if intents is not None and not intents.empty:
                    st.dataframe(intents, hide_index=True, use_container_width=True)
            with d_right:
                st.markdown("**Industry (SIC)**")
                if sic is not None and not sic.empty:
                    st.dataframe(sic, hide_index=True, use_container_width=True)
                st.markdown("**Political**")
                if orientation is not None and not orientation.empty:
                    st.dataframe(orientation, hide_index=True, use_container_width=True)
                if salience is not None and not salience.empty:
                    st.dataframe(salience, hide_index=True, use_container_width=True)
                st.markdown("**Audience & Writing** *(via " + readability_source + ")*")
                if readability_row is not None:
                    st.markdown(
                        f"- Intended Audience: {aud_label} "
                        f"({readability_row['audience_confidence']:.0%} conf)  \n"
                        f"- Writing Maturity: {mat_label} "
                        f"({readability_row['maturity_confidence']:.0%} conf)"
                    )
                st.markdown("**Readability (heuristic)**")
                st.markdown(
                    f"- Grade Level: {readability['grade_level']:.1f}  \n"
                    f"- Reading Ease: {readability['reading_ease']:.1f}  \n"
                    f"- Vocab Diversity: {readability['vocab_diversity']:.2f}"
                )

        st.caption(
            f"Article ID: {article['article_id']}  ·  "
            f"Source: {article['hostname'] or 'Unknown'}  ·  "
            f"Language: {article['language'] or 'Unknown'}"
        )


render_dashboard(dashboard_col)

# ============================================================
# Reader Panel
# ============================================================

if reader_col is not None:
    with reader_col:
        close_col, title_col = st.columns([1, 6])
        with close_col:
            if st.button("X", key="close_reader"):
                st.session_state.show_reader = False
                st.rerun()
        with title_col:
            st.caption(article["title"])
        st.components.v1.html(article_html, height=950, scrolling=True)
