import pandas as pd
import os
from process_article import process_article
from save import save_article_labels

script_dir = os.path.dirname(os.path.abspath(__file__))

CSV = os.path.join(script_dir,"process_data", "articles.csv")

df = pd.read_csv(CSV)

# iterate through csv
for x, row in df.iterrows():
    # pull urls
    url = row['url']
    article_id = process_article(url)

    GT = row['ground_truth_topic']

    if pd.notna(GT):
        save_article_labels(GT, article_id)
    
