from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = PROJECT_ROOT / "data" / "news_analysis.db"

def get_conn():
    return sqlite3.connect(DB_PATH)


def execute(sql, params=None):

    with get_conn() as conn:

        cur = conn.cursor()

        cur.execute(sql, params or [])

        return cur
    
def query(sql, params=None):
    conn = get_conn()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df