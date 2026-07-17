# pipeline/ner.py
from collections import defaultdict

# ent types worht keeping for news analysis
KEEP_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "EVENT",
    "WORK_OF_ART",
}

def extract_ent(doc):
    """
    processes the doc created by spaCy nlp, extracts necessary entities, returns entities/count/label/sentencenum
    """
    entities = {}

    for sent_id, sent in enumerate(doc.sents):

        for ent in sent.ents:

            text = ent.text.strip()
            
            if ent.label_ not in KEEP_LABELS:
                continue
            
            elif text not in entities:

                entities[text] = {
                    "label": ent.label_,
                    "count": 0,
                    "sentences": []
                }

            entities[text]["count"] += 1
            entities[text]["sentences"].append(sent_id)

    return entities