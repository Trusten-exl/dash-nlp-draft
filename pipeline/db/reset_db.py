# db/reset_db.py

from connection import execute

def clear_all_tables():
    """
    self-explanatory
    """

    execute("DELETE FROM article_topics")
    execute("DELETE FROM article_sentiment")
    execute("DELETE FROM entities")
    execute("DELETE FROM entity_sentiment")
    execute("DELETE FROM article_labels")
    execute("DELETE FROM articles")
    execute("DELETE FROM article_intents")
    execute("DELETE FROM article_format")
    execute("DELETE FROM article_sic")
    execute("DELETE FROM political_orientation")
    execute("DELETE FROM political_salience")

    execute("DELETE FROM sqlite_sequence WHERE name='articles'")
    execute("DELETE FROM sqlite_sequence WHERE name='article_topics'")
    execute("DELETE FROM sqlite_sequence WHERE name='article_intents'")
    execute("DELETE FROM sqlite_sequence WHERE name='article_format'")
    execute("DELETE FROM sqlite_sequence WHERE name='entities'")
    execute("DELETE FROM sqlite_sequence WHERE name='entity_sentiment'")
    execute("DELETE FROM sqlite_sequence WHERE name='article_labels'")
    execute("DELETE FROM sqlite_sequence WHERE name='article_sic'")
    execute("DELETE FROM sqlite_sequence WHERE name='political_orientation'")
    execute("DELETE FROM sqlite_sequence WHERE name='political_salience'")
    

    print("All tables cleared.")

clear_all_tables()