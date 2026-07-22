"""
Populate the entity_roles table for every article stored in the DB.

Classifies each PERSON / ORG / EVENT / WORK_OF_ART entity into a specific
role (athlete, actor, musician, politician, executive, sports_team, company,
sporting_event, movie_or_tv_show, or other) via zero-shot MNLI, using the
article text for context, and writes the result to entity_roles. The dashboard reads this to
render the Sports Highlights card (sports articles) or the Important
Entities widget (every other article).

Only entities that get a real role (not "other") and rank in the top
ENRICH_CAP by mention_count get a Wikipedia link (see
entity_roles.classify_entity_roles) — this keeps the number of Claude/
Wikipedia calls bounded regardless of how many articles or entities exist.

Run from the pipeline/ directory on a machine where torch/transformers work:

    cd pipeline
    python reclassify_entities.py
"""

from db.connection import query, get_conn, execute
from entity_roles import classify_entity_roles, CONFIDENCE
from save import save_entity_roles

ENTITY_LABELS = ("PERSON", "ORG", "EVENT", "WORK_OF_ART")


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
            confidence REAL,
            mention_count INTEGER
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(entity_roles)")}
    if "mention_count" not in existing_columns:
        conn.execute("ALTER TABLE entity_roles ADD COLUMN mention_count INTEGER")
    conn.commit()
    conn.close()


def classifiable_article_ids():
    """Every article_id in the DB."""
    df = query("SELECT article_id FROM articles ORDER BY article_id")
    return [int(a) for a in df["article_id"]]


def main():
    ensure_tables()

    ids = classifiable_article_ids()
    total = len(ids)
    print(f"Classifying entities for {total} articles "
          f"(positive-label threshold={CONFIDENCE})...\n")

    placeholders = ", ".join("?" for _ in ENTITY_LABELS)

    for i, aid in enumerate(ids):
        art = query("SELECT text FROM articles WHERE article_id = ?", (aid,))
        if art.empty:
            continue
        article = {"text": art.iloc[0]["text"]}

        entities = query(
            f"SELECT entity_text, entity_label, mention_count FROM entities "
            f"WHERE article_id = ? AND entity_label IN ({placeholders})",
            (aid, *ENTITY_LABELS),
        ).to_dict("records")

        roles = classify_entity_roles(article, entities)
        save_entity_roles(roles, aid)

        matched = [r["entity_text"] for r in roles if r["role"] != "other"]
        print(f"[{i + 1}/{total}] id={aid}  matched={matched or '—'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
