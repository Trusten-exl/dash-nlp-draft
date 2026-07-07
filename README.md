Handwriting a clean, modular version of the news_analysis_pipeline

news_analysis/
│
├── app.py                      # Streamlit entry point (dashboard)
├── requirements.txt
├── .streamlit/
│    └── config.toml
│
├── data/
│    └── news_analysis.db      # SQLite (dev only)
│
├── pipeline/
│    │
│    ├── process_article.py    # MAIN orchestration function
│    ├── ingest.py             # file upload / input handling
│    ├── clean.py              # text cleaning
│    │
│    ├── topics.py            # BART MNLI classification
│    ├── sentiment.py         # VADER or transformer sentiment
│    ├── entities.py          # spaCy NER
│    ├── entity_sentiment.py  # sentence-level aggregation
│    │
│    └── save.py              # ALL database writes
│
├── db/
│    ├── connection.py        # sqlite connection
│    └── queries.py           # reusable SQL queries (optional)
│
├── ui/
│    ├── pages/
│    │    ├── articles.py
│    │    ├── topics.py
│    │    ├── sentiment.py
│    │    ├── entities.py
│    │    └── upload.py      # NEW: upload interface
│    │
│    └── components.py       # reusable UI pieces
│
└── utils/
     ├── text_utils.py
     ├── logging.py
     └── config.py