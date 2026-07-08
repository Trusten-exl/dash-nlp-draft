from transformers import pipeline


# pre-set topic names for use in classification - to be edited
ORIENTATION_LABELS = [
    "Progressive or left-wing",

    "Center-left",

    "Centrist or politically neutral",

    "Center-right",

    "Right-wing or conservative"
]

# ORIENTATION_MAP = {
#     "left-wing or progressive political position": "left",
#     "center-left political position": "center-left",
#     "centrist or politically neutral position": "neutral",
#     "center-right political position": "center-right",
#     "right-wing or conservative political position": "right"
# }

# MAPPED = {v: k for v , k in ORIENTATION_MAP.items()}

# selected model for topic selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_poliical_orientation(article):
    """
    classifies orientation of article using above selected model / topics
    returns dict of orienattion / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=ORIENTATION_LABELS,
        hypothesis_template="The political orientation of this news is {}",
        multi_label=False
    )

    # sort into topics list for use in sql
    orientation = []

    for rank in range(len(result['labels'])):
        orientation.append({
            'orientation': result['labels'][rank],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return orientation



SALIENCE_LABELS = [
    "is highly focused on a political issue, politician, election, government action, or policy debate",
    "contains some meaningful political discussion but politics is not the primary focus",
    "has little or no political relevance and only minor or incidental political references"
]

SALIENCE_MAP = {
    "is highly focused on a political issue, politician, election, government action, or policy debate": "High",
    "contains some meaningful political discussion but politics is not the primary focus": "Medium",
    "has little or no political relevance and only minor or incidental political references": "Low"
}


MAPPED = {v: k for v, k in SALIENCE_MAP.items()}

# selected model for topic selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_article_salience(article):
    """
    classifies topic of article using above selected model / topics
    returns dict of topics / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=SALIENCE_LABELS,
        hypothesis_template="This news article {}",
        multi_label=True
    )

    # sort into topics list for use in sql
    salience = []

    for rank in range(len(result['labels'])):
        salience.append({
            'salience': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return salience

