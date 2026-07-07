# db/save.py

# import helped func for sql execution
from db.connection import execute

def save_article(article):
    """
    sql code for saving articles

    returns article id for use in other tables
    """
    cur = execute("""
    INSERT OR REPLACE INTO articles (

        url,

        title,

        word_count,

        text

    )

    VALUES (?, ?, ?, ?)
    """,
    (
        article['url'],

        article['title'],

        article['word count'],

        article['text']
    ))

    # return article_id for use in connected tables
    article_id = cur.lastrowid

    return article_id

def save_topics(t, article_id):
    """
    sql code for saving article topics
    """
    for x in range(len(t)):
        execute("""
        INSERT INTO article_topics (

            article_id,

            rank,

            topic,

            confidence

        )

        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,

            t[x]['rank'],

            t[x]['topic'],

            t[x]['confidence']
        ))

def save_intents(i, article_id):
    """
    sql code for saving article intent
    """
    for x in range(len(i)):
        execute("""
        INSERT INTO article_intents (

            article_id,

            rank,

            intent,

            confidence

        )

        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,

            i[x]['rank'],

            i[x]['intent'],

            i[x]['confidence']
        ))

def save_formats(f, article_id):
    """
    sql code for saving article format
    """
    for x in range(len(f)):
        execute("""
        INSERT INTO article_format (

            article_id,

            rank,

            format,

            confidence

        )

        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,

            f[x]['rank'],

            f[x]['format'],

            f[x]['confidence']
        ))

def save_sic(d, article_id):
    """
    sql code for saving article sic info
    """

    for level, results in d.items():
        if not results:
            continue

        top = max(results, key=lambda x: x["score"])

        execute("""
        INSERT INTO article_sic (

            article_id,

            level,

            code,

            name,
                
            confidence

        )

        VALUES (?, ?, ?, ?, ?)
        """,
        (
            article_id,
            level,
            top["code"],
            top["name"],
            top["score"]
        ))

def save_article_sentiment(sentiment, article_id):
    """
    save full article sentiment to db
    """
    execute("""
    INSERT OR REPLACE INTO
    article_sentiment
    
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (
        article_id,

        sentiment['score'],

        sentiment['sentence count'],
        
        sentiment['positive sentences'],

        sentiment['neutral sentences'],
        
        sentiment['negative sentences']
    ))

# save entities to db **SHOULD THIS RETURN ENTITY ID FOR PERSISTENCE? OR JUST TALE SENTIMENT?
def save_entities(entities, article_id):
    """
    save all entities to db
    """
    for entity_text, info in entities.items():
        execute("""
        INSERT INTO entities (

                    article_id,

                    entity_text,

                    entity_label,

                    mention_count

                )

                VALUES (?, ?, ?, ?)
                """,
                (
                    article_id,

                    entity_text,

                    info['label'],

                    info['count']
                ))
        

def save_entity_sentiment(ent_sent, article_id):
    """
    sql code to save entity level sentiment values to db
    """
    for entity, (score, sent_count) in ent_sent.items():
        execute("""
        INSERT INTO entity_sentiment(
                article_id,

                entity_text,

                sentiment,

                sentence_count
                
            )

            VALUES (?, ?, ?, ?)
            """,
        (
            article_id,

            entity,

            score,

            sent_count
        ))

# save article_labels to db
def save_article_labels(gt, article_id):
    """"
    saving ground truth labels if there
    """
    execute("""
    INSERT INTO article_labels(
            article_id,
            
            label_topic
            
            )
            
            VALUES (?, ?)
            """,
        (
            article_id,

            gt
        ))
    

#  politics
def save_p_orientation(p, article_id):
    
    """
    sql code for saving political_orientation
    """
    for x in range(len(p)):
        execute("""
        INSERT INTO political_orientation (

            article_id,

            rank,

            orientation,

            confidence

        )

        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,

            p[x]['rank'],

            p[x]['orientation'],

            p[x]['confidence']
        ))


def save_p_salience(p, article_id):
    
    """
    sql code for saving political_salience
    """
    for x in range(len(p)):
        execute("""
        INSERT INTO political_salience (

            article_id,

            rank,

            salience,

            confidence

        )

        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,

            p[x]['rank'],

            p[x]['salience'],

            p[x]['confidence']
        ))