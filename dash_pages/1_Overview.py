# ============================================================
# 1_Overview.py — Corpus Intelligence (single-screen overview)
# ============================================================

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from pipeline.db.connection import query


# ============================================================
# Page Config (guarded: harmless whether run standalone or via st.navigation)
# ============================================================

try:
    st.set_page_config(page_title="Corpus Intelligence", page_icon="📊", layout="wide")
except Exception:
    pass

st.markdown(
    """
<style>
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 0.75rem;
    max-width: 1700px;
}
[data-testid="stAppViewContainer"] { overflow-y: auto; }
[data-testid="stHeader"] { z-index: 999; }
[data-testid="stVerticalBlock"] { gap: 0.7rem; }
[data-testid="stImage"] { margin-bottom: 0; }
hr { margin: 0.6rem 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Small shared helpers (kept local to avoid coupling to the detail page)
# ============================================================


def section_label_html(text: str) -> str:
    return (
        '<div style="font-size:.62rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:.05rem;margin-bottom:6px;">{text}</div>'
    )


def small_card_html(label: str, value: str, color: str = "#888",
                    text_color: str = "#000000", subtitle: str = "") -> str:
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


def sentiment_color(score: float) -> str:
    if score >= 0.1:
        return "#1b9e4b"
    if score <= -0.1:
        return "#d93025"
    return "#7a7a7a"


def orientation_color(label: str) -> str:
    label = str(label).lower()
    if "left" in label or "progressive" in label:
        return "#2d6cdf"
    if "right" in label or "conservative" in label:
        return "#d93025"
    return "#7a7a7a"


LEAN_SHORT = {
    "Progressive or left-wing": "Left",
    "Center-left": "Center-left",
    "Centrist or politically neutral": "Neutral",
    "Center-right": "Center-right",
    "Right-wing or conservative": "Right",
}


# ============================================================
# Readability / writing maturity (per article → averaged per author)
# ============================================================


def _count_syllables(word: str) -> int:
    vowels = re.findall(r"[aeiou]+", word.lower())
    count = len(vowels)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def maturity_score(text: str) -> float:
    """0–100 writing-complexity score (higher = more sophisticated prose)."""
    if not text:
        return 50.0
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sent = max(len(sentences), 1)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    n_words = max(len(words), 1)
    n_syl = sum(_count_syllables(w) for w in words)
    vocab_diversity = len(set(w.lower() for w in words)) / n_words
    ease = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syl / n_words)
    ease = max(0.0, min(100.0, ease))
    complexity = (100 - ease) / 100
    return max(0.0, min(100.0, (complexity * 0.6 + vocab_diversity * 0.4) * 100))


def maturity_label(score: float) -> str:
    if score < 25:
        return "Accessible"
    if score < 50:
        return "Standard"
    if score < 75:
        return "Sophisticated"
    return "Advanced"


def maturity_color(score: float) -> str:
    if score < 25:
        return "#2ca02c"
    if score < 50:
        return "#4a9eff"
    if score < 75:
        return "#9467bd"
    return "#d62728"


# ============================================================
# Data loaders
# ============================================================


@st.cache_data(ttl=300)
def load_data():
    articles = query("SELECT * FROM articles")
    sentiment = query(
        "SELECT article_id, sentiment_score FROM article_sentiment"
    )
    topic1 = query(
        "SELECT article_id, topic FROM article_topics WHERE rank = 1"
    )
    orient1 = query(
        "SELECT article_id, orientation FROM political_orientation WHERE rank = 1"
    )
    salience1 = query(
        "SELECT article_id, salience FROM political_salience WHERE rank = 1"
    )
    # Per-article entity data (kept un-aggregated so the cloud can be filtered by topic).
    entities_raw = query(
        "SELECT article_id, entity_text, mention_count FROM entities"
    )
    entity_sent_raw = query(
        "SELECT article_id, entity_text, sentiment FROM entity_sentiment"
    )
    # Writing-maturity score per article: prefer the NLP-classified score,
    # fall back to the heuristic when an article hasn't been classified.
    try:
        rd = query("SELECT article_id, maturity_score FROM article_readability")
        db_maturity = dict(zip(rd["article_id"], rd["maturity_score"]))
    except Exception:
        db_maturity = {}  # table not created yet
    maturity_map = {
        int(r["article_id"]): (
            float(db_maturity[r["article_id"]])
            if r["article_id"] in db_maturity
            else maturity_score(r["text"] or "")
        )
        for _, r in articles.iterrows()
    }
    return (
        articles, sentiment, topic1, orient1, salience1,
        entities_raw, entity_sent_raw, maturity_map,
    )


(
    articles, sentiment, topic1, orient1, salience1,
    entities_raw, entity_sent_raw, maturity_map,
) = load_data()

if articles.empty:
    st.error("No articles found.")
    st.stop()


# ============================================================
# Author cleaning / intelligence
# ============================================================

_AUTHOR_JUNK = {
    "updated", "the politics desk", "staff", "editor", "editors",
    "leer en español", "leer en espanol",
}


def split_authors(raw) -> list[str]:
    """Split a messy multi-author string into individual, de-junked names."""
    if not raw:
        return []
    out = []
    for part in re.split(r"[;,]| and ", str(raw)):
        p = part.strip()
        if not p or "�" in p:            # drop decode-artifact tokens
            continue
        low = p.lower()
        # "Leer En Español" / "Read In ..." are localisation links, not authors.
        if low in _AUTHOR_JUNK or low.startswith(("updated", "leer ", "read in")):
            continue
        if len(p) < 4 or " " not in p:        # require a plausible full name
            continue
        out.append(p)
    return out


def build_author_records(articles, sentiment, orient1, topic1, maturity_map):
    sent_map = dict(zip(sentiment["article_id"], sentiment["sentiment_score"]))
    orient_map = dict(zip(orient1["article_id"], orient1["orientation"]))
    topic_map = dict(zip(topic1["article_id"], topic1["topic"]))

    authors: dict[str, set] = {}
    for _, r in articles.iterrows():
        for name in split_authors(r["author"]):
            authors.setdefault(name, set()).add(r["article_id"])

    records = []
    for name, ids in authors.items():
        ids = list(ids)
        sents = [sent_map[i] for i in ids if i in sent_map]
        orients = [orient_map[i] for i in ids if i in orient_map]
        topics = [topic_map[i] for i in ids if i in topic_map]
        maturities = [maturity_map[i] for i in ids if i in maturity_map]
        avg_mat = (sum(maturities) / len(maturities)) if maturities else 0.0
        records.append(
            {
                "author": name,
                "n": len(ids),
                "avg_sent": (sum(sents) / len(sents)) if sents else 0.0,
                "orientation": Counter(orients).most_common(1)[0][0] if orients else "N/A",
                "top_topic": Counter(topics).most_common(1)[0][0] if topics else "—",
                "maturity": avg_mat,
            }
        )

    records.sort(key=lambda d: (d["n"], abs(d["avg_sent"])), reverse=True)
    return records


def author_panel_html(records, limit=5) -> str:
    if not records:
        return '<div style="color:#888;font-size:.8rem;">No attributable authors.</div>'

    rows = ""
    for i, rec in enumerate(records[:limit], start=1):
        oc = orientation_color(rec["orientation"])
        lean = LEAN_SHORT.get(rec["orientation"], "—")
        sc = sentiment_color(rec["avg_sent"])
        mc = maturity_color(rec["maturity"])
        mlabel = maturity_label(rec["maturity"])
        star = (
            ' <span title="Multiple stories" style="color:#e0a800;">★</span>'
            if rec["n"] >= 2
            else ""
        )
        rows += f"""
<div style="display:flex;align-items:center;gap:9px;padding:5px 2px;
            border-bottom:1px solid rgba(120,120,120,.15);">
  <div style="min-width:18px;text-align:right;color:#999;font-weight:700;font-size:.8rem;">{i}</div>
  <div style="flex:1;min-width:0;">
    <div style="font-size:.86rem;font-weight:600;color:#222;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{rec['author']}{star}</div>
    <div style="font-size:.6rem;color:#888;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;">{rec['top_topic']}</div>
  </div>
  <div style="text-align:center;min-width:30px;">
    <div style="font-size:.55rem;color:#888;text-transform:uppercase;">Stories</div>
    <div style="font-size:.85rem;font-weight:700;color:#222;">{rec['n']}</div>
  </div>
  <div style="text-align:center;min-width:48px;">
    <div style="font-size:.55rem;color:#888;text-transform:uppercase;">Sentiment</div>
    <div style="font-size:.85rem;font-weight:700;color:{sc};">{rec['avg_sent']:+.2f}</div>
  </div>
  <div style="text-align:center;min-width:78px;">
    <div style="font-size:.55rem;color:#888;text-transform:uppercase;">Writing</div>
    <div style="font-size:.72rem;font-weight:700;color:{mc};"
         title="Writing maturity {rec['maturity']:.0f}/100">{mlabel}</div>
  </div>
  <div style="min-width:72px;text-align:right;">
    <span style="background:{oc}22;border:1px solid {oc};color:{oc};
                 padding:2px 8px;border-radius:999px;font-size:.6rem;white-space:nowrap;">{lean}</span>
  </div>
</div>"""

    return (
        '<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);'
        f'padding:8px 12px;background:rgba(0,0,0,.02);">{rows}</div>'
    )


# ============================================================
# Corpus word cloud
# ============================================================


def build_corpus_wordcloud(entities_raw, entity_sent_raw, article_ids=None, max_words=90):
    if entities_raw is None or entities_raw.empty:
        return None

    ent = entities_raw
    esent = entity_sent_raw
    if article_ids is not None:
        ent = ent[ent["article_id"].isin(article_ids)]
        esent = esent[esent["article_id"].isin(article_ids)]
    if ent.empty:
        return None

    # Aggregate across the (optionally filtered) articles.
    mentions = ent.groupby("entity_text", as_index=False)["mention_count"].sum()
    mentions = mentions.rename(columns={"mention_count": "mentions"})
    sent = esent.groupby("entity_text", as_index=False)["sentiment"].mean()

    merged = mentions.merge(sent, on="entity_text", how="left")
    merged["mentions"] = pd.to_numeric(merged["mentions"], errors="coerce").fillna(1).astype(int)
    merged["sentiment"] = pd.to_numeric(merged["sentiment"], errors="coerce").fillna(0.0)
    merged = merged[merged["entity_text"].astype(str).str.len() > 1]
    merged = merged[merged["mentions"] > 0]
    if merged.empty:
        return None

    merged = merged.sort_values("mentions", ascending=False).head(max_words)

    freq = dict(zip(merged["entity_text"], merged["mentions"]))
    sentiment_map = dict(zip(merged["entity_text"], merged["sentiment"]))
    max_abs = max(0.3, float(merged["sentiment"].abs().max()))

    def _blend(pale, dark, t):
        return tuple(int(round(p + (d - p) * t)) for p, d in zip(pale, dark))

    def color_func(word, **kwargs):
        score = sentiment_map.get(word, 0.0)
        if -0.1 < score < 0.1:
            return "rgb(150,150,150)"
        t = min(1.0, abs(score) / max_abs)
        if score > 0:
            r, g, b = _blend((150, 205, 165), (12, 105, 45), t)
        else:
            r, g, b = _blend((225, 165, 160), (150, 25, 20), t)
        return f"rgb({r},{g},{b})"

    wc = WordCloud(
        width=1000,
        height=440,
        background_color=None,
        mode="RGBA",
        color_func=color_func,
        prefer_horizontal=0.85,
        max_words=max_words,
        collocations=False,
    ).generate_from_frequencies(freq)

    return wc.to_image()


# ============================================================
# Political profile of the corpus (segmented spectrum bars, no charts)
# ============================================================

_ORIENT_ORDER = [
    "Progressive or left-wing",
    "Center-left",
    "Centrist or politically neutral",
    "Center-right",
    "Right-wing or conservative",
]
_ORIENT_COLORS = {
    "Progressive or left-wing": "#2F6BFF",
    "Center-left": "#8CB5FF",
    "Centrist or politically neutral": "#CFCFCF",
    "Center-right": "#F4A6A6",
    "Right-wing or conservative": "#D62828",
}
_ORIENT_SCORE = {
    "Progressive or left-wing": 10,
    "Center-left": 30,
    "Centrist or politically neutral": 50,
    "Center-right": 70,
    "Right-wing or conservative": 90,
}
_SALIENCE_ORDER = ["Low", "Medium", "High"]
_SALIENCE_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


def _segmented_bar(order, colors, counts) -> str:
    total = sum(counts.get(k, 0) for k in order) or 1
    segs = ""
    for k in order:
        n = counts.get(k, 0)
        if n == 0:
            continue
        pct = n / total * 100
        label = f"{n}" if pct >= 8 else ""
        segs += (
            f'<div title="{k}: {n}" style="width:{pct:.1f}%;background:{colors[k]};'
            'height:100%;display:flex;align-items:center;justify-content:center;'
            f'font-size:.62rem;font-weight:700;color:#fff;">{label}</div>'
        )
    return (
        '<div style="display:flex;height:24px;border-radius:6px;overflow:hidden;'
        f'border:1px solid rgba(120,120,120,.2);">{segs}</div>'
    )


def _legend(order, colors, counts) -> str:
    items = ""
    for k in order:
        n = counts.get(k, 0)
        if n == 0:
            continue
        short = LEAN_SHORT.get(k, k)
        items += (
            '<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;'
            'font-size:.62rem;color:#666;">'
            f'<span style="width:9px;height:9px;border-radius:2px;background:{colors[k]};'
            'display:inline-block;"></span>'
            f'{short} · {n}</span>'
        )
    return f'<div style="margin-top:5px;line-height:1.6;">{items}</div>'


def political_profile_html(orient1, salience1) -> str:
    o_counts = orient1["orientation"].value_counts().to_dict()
    s_counts = salience1["inclination"].value_counts().to_dict()

    total_o = sum(o_counts.get(k, 0) for k in _ORIENT_ORDER) or 1
    avg = sum(_ORIENT_SCORE[k] * o_counts.get(k, 0) for k in _ORIENT_ORDER) / total_o
    if avg < 20:
        avg_lean, avg_col = "Left", "#2F6BFF"
    elif avg < 40:
        avg_lean, avg_col = "Center-left", "#5B8DEF"
    elif avg < 60:
        avg_lean, avg_col = "Neutral", "#7a7a7a"
    elif avg < 80:
        avg_lean, avg_col = "Center-right", "#E07A7A"
    else:
        avg_lean, avg_col = "Right", "#D62828"

    return f"""
<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);
            padding:12px 14px;background:rgba(0,0,0,.02);">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <div style="font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.04rem;">Orientation</div>
    <div style="font-size:.7rem;color:#888;">Corpus leans
        <span style="font-weight:700;color:{avg_col};">{avg_lean}</span></div>
  </div>
  <div style="margin-top:5px;">{_segmented_bar(_ORIENT_ORDER, _ORIENT_COLORS, o_counts)}</div>
  {_legend(_ORIENT_ORDER, _ORIENT_COLORS, o_counts)}

  <div style="font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.04rem;
              margin-top:14px;">Salience</div>
  <div style="margin-top:5px;">{_segmented_bar(_SALIENCE_ORDER, _SALIENCE_COLORS, s_counts)}</div>
  {_legend(_SALIENCE_ORDER, _SALIENCE_COLORS, s_counts)}
</div>
"""


# ============================================================
# Topic landscape treemap (sized by volume, coloured by sentiment)
# ============================================================


def build_topic_treemap(topic1, sentiment):
    if topic1 is None or topic1.empty:
        return None

    df = topic1.merge(
        sentiment[["article_id", "sentiment_score"]], on="article_id", how="left"
    )
    g = (
        df.groupby("topic")
        .agg(n=("article_id", "count"), sent=("sentiment_score", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    g["sent"] = g["sent"].fillna(0.0)
    if g.empty:
        return None

    bound = max(0.3, float(g["sent"].abs().max()))
    fig = go.Figure(
        go.Treemap(
            labels=g["topic"],
            parents=[""] * len(g),
            values=g["n"],
            marker=dict(
                colors=g["sent"],
                colorscale=[[0.0, "#d93025"], [0.5, "#d9d9d9"], [1.0, "#1b9e4b"]],
                cmid=0,
                cmin=-bound,
                cmax=bound,
                line=dict(width=1, color="rgba(255,255,255,.35)"),
            ),
            textinfo="label+value",
            texttemplate="<b>%{label}</b><br>%{value} stories",
            hovertemplate="%{label}<br>%{value} stories<br>avg sentiment %{color:.2f}<extra></extra>",
            tiling=dict(pad=2),
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=4, r=4, t=6, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#222", size=12),
    )
    return fig


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div style="font-size:1.6rem;font-weight:800;line-height:1.1;">Corpus Intelligence</div>'
    '<div style="font-size:.8rem;color:#888;margin-bottom:.3rem;">'
    "Aggregate trends across the full article collection</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# ROW 1: KPI strip
# ============================================================

author_records = build_author_records(articles, sentiment, orient1, topic1, maturity_map)
n_authors = len(author_records)
n_multi = sum(1 for r in author_records if r["n"] >= 2)

n_sources = articles["sitename"].fillna(articles["hostname"]).dropna().nunique()

avg_sent = float(sentiment["sentiment_score"].mean()) if not sentiment.empty else 0.0
avg_words = int(articles["word_count"].fillna(0).mean())

dts = pd.to_datetime(articles["publish_date"], errors="coerce").dropna()
if not dts.empty:
    span = f"{dts.min():%b %Y} – {dts.max():%b %Y}"
else:
    span = "N/A"

kpis = [
    ("Articles", f"{len(articles):,}", "#4a9eff", "#000000", ""),
    ("Authors", f"{n_authors}", "#6f42c1", "#000000", f"{n_multi} with 2+ stories"),
    ("Sources", f"{n_sources}", "#fd7e14", "#000000", ""),
    ("Coverage Span", span, "#888", "#000000", f"{len(dts)} dated"),
    ("Avg Sentiment", f"{avg_sent:+.2f}", sentiment_color(avg_sent), sentiment_color(avg_sent), ""),
    ("Avg Length", f"{avg_words:,} words", "#888", "#000000", ""),
]

for col, (label, value, color, tcolor, sub) in zip(st.columns(6), kpis):
    with col:
        st.markdown(small_card_html(label, value, color, tcolor, sub), unsafe_allow_html=True)

st.markdown("<div style='margin-top:2px;'></div>", unsafe_allow_html=True)


# ============================================================
# ROW 2: 2×2 graphics grid
#   ┌──────────────────────┬──────────────────────┐
#   │ Corpus Entity Cloud  │ Author Intelligence  │
#   ├──────────────────────┼──────────────────────┤
#   │ Sentiment Timeline   │ Topic Landscape      │
#   └──────────────────────┴──────────────────────┘
# ============================================================

top_left, top_right = st.columns(2, gap="large")

with top_left:
    st.markdown(section_label_html("Corpus Entity Cloud"), unsafe_allow_html=True)

    # Topic filter: order options by story count (matches the treemap).
    topic_order = topic1["topic"].value_counts().index.tolist()
    topic_choice = st.selectbox(
        "Filter entities by topic",
        ["All topics"] + topic_order,
        label_visibility="collapsed",
        key="wc_topic_filter",
    )
    if topic_choice == "All topics":
        filter_ids = None
    else:
        filter_ids = set(
            topic1.loc[topic1["topic"] == topic_choice, "article_id"].tolist()
        )

    wc_image = build_corpus_wordcloud(entities_raw, entity_sent_raw, filter_ids)
    if wc_image is not None:
        st.image(wc_image, use_container_width=True)
    else:
        st.info("No entities for this topic.")

with top_right:
    st.markdown(
        section_label_html(f"Author Intelligence · {n_multi} recurring"),
        unsafe_allow_html=True,
    )
    st.markdown(author_panel_html(author_records), unsafe_allow_html=True)

st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns(2, gap="large")

with bottom_left:
    st.markdown(section_label_html("Political Profile"), unsafe_allow_html=True)
    if orient1.empty and salience1.empty:
        st.info("No political data available.")
    else:
        st.markdown(political_profile_html(orient1, salience1), unsafe_allow_html=True)

with bottom_right:
    st.markdown(section_label_html("Topic Landscape"), unsafe_allow_html=True)
    tm_fig = build_topic_treemap(topic1, sentiment)
    if tm_fig is not None:
        st.plotly_chart(tm_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No topic data available.")

st.caption(
    "Entity cloud size = total mentions, colour = mention-weighted sentiment.  ·  "
    "Treemap size = stories, colour = average sentiment (red → green).  ·  "
    "Political bars = share of articles per category.  ·  "
    "Writing = avg text-complexity of an author's stories.  ·  "
    "★ marks authors with more than one story."
)
