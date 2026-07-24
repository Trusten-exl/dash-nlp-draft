"""
Run the pipeline against every URL in articles_claude.csv (scraping only the
ones not already in the DB) and compare predictions to its ground-truth
columns. Writes one row per article to gt_comparison.csv for manual review,
plus a per-field accuracy summary printed to stdout.

Run from the pipeline/ directory on a machine where torch/transformers work:

    python compare_to_gt.py
"""

import os
import pandas as pd

from db.connection import query
from process_article import process_article
from entity_roles import classify_entity_roles
from save import save_entity_roles

script_dir = os.path.dirname(os.path.abspath(__file__))
GT_CSV = os.path.join(script_dir, "process_data", "articles_claude.csv")
OUT_CSV = os.path.join(script_dir, "process_data", "gt_comparison.csv")

ENTITY_ROLE_LABELS = ("PERSON", "ORG", "EVENT", "WORK_OF_ART")

# (display name, predictions table, predictions column, GT column) for every
# rank-based zero-shot field. Audience/maturity/sentiment/entities are
# special-shaped and handled separately in compare_row.
FIELDS = [
    ("topic", "article_topics", "topic", "ground_truth_topic"),
    ("intent", "article_intents", "intent", "ground_truth_intent"),
    ("format", "article_format", "format", "ground_truth_format"),
    ("orientation", "political_orientation", "orientation", "ground_truth_orientation"),
    ("salience", "political_salience", "salience", "ground_truth_salience"),
]


def _norm(s):
    return str(s).strip().lower() if pd.notna(s) else ""


def _top(table, col, article_id):
    df = query(f"SELECT {col} FROM {table} WHERE article_id = ? AND rank = 1", (article_id,))
    return df.iloc[0][col] if not df.empty else None


# Thresholds on article_sentiment.sentiment_score (avg of per-sentence scores,
# positive/negative/0). Plurality-of-sentence-counts was tried first and only
# hit 48% - most articles have a majority of purely-factual (neutral)
# sentences even when clearly positive/negative overall, so it almost always
# picked Neutral. The continuous score separates the GT classes cleanly
# (median -0.23 / 0.05 / 0.17 for Negative/Neutral/Positive); a symmetric
# +-0.1 band around 0 gets 62% on the GT set.
SENTIMENT_BAND = 0.1


def _sentiment_label(article_id):
    df = query(
        "SELECT sentiment_score FROM article_sentiment WHERE article_id = ?",
        (article_id,),
    )
    if df.empty:
        return None
    score = df.iloc[0]["sentiment_score"]
    if score <= -SENTIMENT_BAND:
        return "Negative"
    if score >= SENTIMENT_BAND:
        return "Positive"
    return "Neutral"


def parse_gt_entities(raw):
    """'Name:role;Name:role' -> {name_lower: role_lower}"""
    if pd.isna(raw) or not str(raw).strip():
        return {}
    pairs = {}
    for chunk in str(raw).split(";"):
        if ":" not in chunk:
            continue
        name, role = chunk.rsplit(":", 1)
        pairs[name.strip().lower()] = role.strip().lower()
    return pairs


def entity_scores(article_id, gt_raw):
    """
    Exact-name (case-insensitive) precision/recall of predicted entity roles
    vs GT. ponytail: no fuzzy name matching (e.g. "Nvidia" vs "Nvidia Corp"
    won't match) - add if the notes column shows that being the dominant miss.
    """
    gt = parse_gt_entities(gt_raw)
    preds = query(
        "SELECT entity_text, role FROM entity_roles WHERE article_id = ? AND role != 'other'",
        (article_id,),
    )
    pred = {r["entity_text"].strip().lower(): r["role"] for _, r in preds.iterrows()}

    matched = sum(1 for name, role in gt.items() if pred.get(name) == role)
    precision = (matched / len(pred)) if pred else None
    recall = (matched / len(gt)) if gt else None

    notes = []
    for name, role in gt.items():
        if name not in pred:
            notes.append(f"missed:{name}({role})")
        elif pred[name] != role:
            notes.append(f"wrong_role:{name}(gt={role},pred={pred[name]})")
    for name, role in pred.items():
        if name not in gt:
            notes.append(f"extra:{name}({role})")

    return precision, recall, "; ".join(notes)


def ensure_processed(url):
    """Return the article_id for this url, scraping it if it isn't in the DB yet."""
    existing = query("SELECT article_id FROM articles WHERE url = ?", (url,))
    if not existing.empty:
        return int(existing.iloc[0]["article_id"])
    return process_article(url)


def ensure_entities_classified(article_id):
    if not query("SELECT 1 FROM entity_roles WHERE article_id = ?", (article_id,)).empty:
        return
    art = query("SELECT text FROM articles WHERE article_id = ?", (article_id,))
    if art.empty:
        return
    placeholders = ", ".join("?" for _ in ENTITY_ROLE_LABELS)
    entities = query(
        f"SELECT entity_text, entity_label, mention_count FROM entities "
        f"WHERE article_id = ? AND entity_label IN ({placeholders})",
        (article_id, *ENTITY_ROLE_LABELS),
    ).to_dict("records")
    roles = classify_entity_roles({"text": art.iloc[0]["text"]}, entities)
    save_entity_roles(roles, article_id)


def compare_row(row, article_id):
    result = {"article_id": article_id, "url": row["url"]}

    for name, table, col, gt_col in FIELDS:
        pred = _top(table, col, article_id)
        result[f"{name}_gt"] = row[gt_col]
        result[f"{name}_pred"] = pred
        result[f"{name}_match"] = _norm(pred) == _norm(row[gt_col])

    read = query(
        "SELECT audience_label, maturity_label FROM article_readability WHERE article_id = ?",
        (article_id,),
    )
    audience_pred = read.iloc[0]["audience_label"] if not read.empty else None
    maturity_pred = read.iloc[0]["maturity_label"] if not read.empty else None
    result["audience_gt"] = row["ground_truth_audience"]
    result["audience_pred"] = audience_pred
    result["audience_match"] = _norm(audience_pred) == _norm(row["ground_truth_audience"])
    result["maturity_gt"] = row["ground_truth_maturity"]
    result["maturity_pred"] = maturity_pred
    result["maturity_match"] = _norm(maturity_pred) == _norm(row["ground_truth_maturity"])

    sentiment_pred = _sentiment_label(article_id)
    result["sentiment_gt"] = row["ground_truth_sentiment"]
    result["sentiment_pred"] = sentiment_pred
    result["sentiment_match"] = _norm(sentiment_pred) == _norm(row["ground_truth_sentiment"])

    precision, recall, notes = entity_scores(article_id, row["ground_truth_entities"])
    result["entity_precision"] = precision
    result["entity_recall"] = recall
    result["entity_notes"] = notes

    return result


MATCH_COLS = [f"{name}_match" for name, *_ in FIELDS] + [
    "audience_match", "maturity_match", "sentiment_match",
]


def main():
    gt = pd.read_csv(GT_CSV)
    rows = []

    for i, row in gt.iterrows():
        print(f"[{i + 1}/{len(gt)}] {row['url']}")
        article_id = ensure_processed(row["url"])
        if article_id is None:
            print("  skipped (scrape failed)")
            continue
        ensure_entities_classified(article_id)
        rows.append(compare_row(row, article_id))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(out)} rows to {OUT_CSV}\n")
    print("Accuracy by field:")
    for col in MATCH_COLS:
        name = col.removesuffix("_match")
        print(f"  {name:12s} {out[col].mean():.0%}  ({int(out[col].sum())}/{len(out)})")
    print(
        f"  entities     "
        f"precision={out['entity_precision'].mean():.0%}  "
        f"recall={out['entity_recall'].mean():.0%}"
    )


if __name__ == "__main__":
    main()
