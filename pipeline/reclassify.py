"""
Re-run SIC classification (with the new industry-relevance gate) and the
audience / writing-maturity NLP over the articles ALREADY stored in the DB,
updating rows in place.

Why not process_article?  process_article re-scrapes a URL and inserts a new
article row (duplicating the corpus) and re-runs every stage. This script
touches only article_sic + article_readability and reuses the stored text.

Run from the pipeline/ directory on a machine where torch/transformers work:

    cd pipeline
    python reclassify.py
"""

from db.connection import query, execute, get_conn
from sic import classify_sic_article
from readability import classify_readability
from save import save_sic, save_readability


def ensure_tables():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS article_readability (
            article_id INTEGER PRIMARY KEY,
            audience_label TEXT,
            audience_score REAL,
            audience_confidence REAL,
            maturity_label TEXT,
            maturity_score REAL,
            maturity_confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()


def main():
    ensure_tables()

    articles = query(
        "SELECT article_id, title, description, text FROM articles ORDER BY article_id"
    )
    total = len(articles)
    print(f"Reclassifying {total} articles...\n")

    kept = 0
    for i, row in articles.iterrows():
        aid = int(row["article_id"])
        article = {
            "title": row["title"],
            "description": row["description"],
            "text": row["text"],
        }

        # --- SIC: clear stale rows, then save only if industry-related ---
        execute("DELETE FROM article_sic WHERE article_id = ?", (aid,))
        sic = classify_sic_article(article)
        if sic["related"]:
            save_sic(sic["predictions"], aid)
            kept += 1
            div = sic["predictions"]["division"][0]
            sic_str = f'{div["name"]} ({div["score"]:.2f})'
        else:
            td = sic.get("top_division")
            best = f', best={td["name"]} {td["score"]:.2f}' if td else ""
            sic_str = (
                f'skipped [{sic.get("reason", "?")}] '
                f'P(industry)={sic["relevance_score"]:.2f}{best}'
            )

        # --- Audience / maturity: upsert ---
        rd = classify_readability(article)
        save_readability(rd, aid)

        print(
            f"[{i + 1}/{total}] id={aid}  SIC: {sic_str}  |  "
            f'audience={rd["audience"]["label"]}  maturity={rd["maturity"]["label"]}'
        )

    print(f"\nDone. SIC saved for {kept}/{total} articles; "
          f"{total - kept} gated out as non-industry.")


if __name__ == "__main__":
    main()
