"""
Populate the entity_roles table for sports articles ALREADY stored in the DB.

For every article whose top topic is Sports, this classifies each PERSON /
EVENT entity as an athlete / sporting_event / other (via zero-shot MNLI, using
the article text for context) and writes the result to entity_roles. The
article-detail "Sports Highlights" card reads this to list real athletes and
games instead of every named entity.

Only sports articles are processed because that's the only place the card
renders; this keeps the number of model calls small.

Run from the pipeline/ directory on a machine where torch/transformers work:

    cd pipeline
    python reclassify_entities.py
"""

from db.connection import query, get_conn, execute
from entity_roles import classify_sports_entities, CONFIDENCE
from save import save_entity_roles

def clear_roles():
    execute("DELETE FROM sqlite_sequence WHERE name='entity_roles'")
    execute("DELETE FROM entity_roles")


def ensure_tables():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            entity_text TEXT,
            entity_label TEXT,
            role TEXT,
            url TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()


def sports_article_ids():
    """article_ids whose highest-confidence topic is Sports."""
    df = query(
        """
        SELECT article_id
        FROM (
            SELECT article_id, topic,
                   ROW_NUMBER() OVER (
                       PARTITION BY article_id ORDER BY confidence DESC
                   ) AS rn
            FROM article_topics
        )
        WHERE rn = 1 AND LOWER(TRIM(topic)) = 'sports'
        ORDER BY article_id
        """
    )
    return [int(a) for a in df["article_id"]]

def main():
    ensure_tables()

    ids = sports_article_ids()
    total = len(ids)
    print(f"Classifying entities for {total} sports articles "
          f"(positive-label threshold={CONFIDENCE})...\n")

    for i, aid in enumerate(ids):
        art = query("SELECT text FROM articles WHERE article_id = ?", (aid,))
        if art.empty:
            continue
        article = {"text": art.iloc[0]["text"]}

        entities = query(
            "SELECT entity_text, entity_label FROM entities "
            "WHERE article_id = ? AND entity_label IN ('PERSON', 'EVENT')",
            (aid,),
        ).to_dict("records")

        roles = classify_sports_entities(article, entities)
        save_entity_roles(roles, aid)

        athletes = [r["entity_text"] for r in roles if r["role"] == "athlete"]
        events = [r["entity_text"] for r in roles if r["role"] == "sporting_event"]
        print(
            f"[{i + 1}/{total}] id={aid}  "
            f"athletes={athletes or '—'}  events={events or '—'}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
