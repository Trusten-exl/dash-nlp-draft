# Create db tables
from connection import get_conn

conn = get_conn()
cursor = conn.cursor()


# Article level data
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (

    article_id INTEGER PRIMARY KEY AUTOINCREMENT,

    url TEXT NOT NULL,

    hostname TEXT,
    sitename TEXT,

    title TEXT,
    description TEXT,

    author TEXT,

    publish_date TEXT,
    modified_date TEXT,

    language TEXT,

    image_url TEXT,

    word_count INTEGER,
    text TEXT
)
""")

# article topics
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_topics (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    rank INTEGER,

    topic TEXT,

    confidence REAL
)
""")

# article intents
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_intents (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    rank INTEGER,

    intent TEXT,

    confidence REAL
)
""")

# article sic
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_sic (
               
    id INTEGER PRIMARY KEY AUTOINCREMENT,
               
    article_id INTEGER NOT NULL,
               
    level TEXT NOT NULL,
               
    code TEXT NOT NULL,
               
    name TEXT NOT NULL,
               
    confidence REAL NOT NULL
)
""")

# article format
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_format (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    rank INTEGER,

    format TEXT,

    confidence REAL
)
""")

# article labels
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_labels(
    
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    article_id INTEGER,
    
    label_topic TEXT
)
""")

# entity per article
cursor.execute("""
CREATE TABLE IF NOT EXISTS entities (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    entity_text TEXT,

    entity_label TEXT,

    mention_count INTEGER
)
""")


# full article sentiment tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS article_sentiment (

    article_id INTEGER PRIMARY KEY,

    sentiment_score REAL,

    sentence_count REAL,
    
    positive REAL,

    neutral REAL,

    negative REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS sent_sentiment (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    article_id INTEGER,

    sentence TEXT,

    sentiment TEXT,
    
    score REAL
)
""")

# sentiment per entity
cursor.execute("""
CREATE TABLE IF NOT EXISTS entity_sentiment (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    entity_text TEXT,

    sentiment REAL,

    sentence_count INTEGER
)
""")

# politics
cursor.execute("""
CREATE TABLE IF NOT EXISTS political_orientation (
               
    id INTEGER PRIMARY KEY AUTOINCREMENT,
               
    article_id INTEGER,
               
    rank INTEGER,
               
    orientation TEXT,
               
    confidence REAL               
)               
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS political_salience (
               
    id INTEGER PRIMARY KEY AUTOINCREMENT,
               
    article_id INTEGER,
               
    rank  INTEGER,
               
    salience TEXT,
               
    confidence REAL               
)               
""")
conn.commit()
