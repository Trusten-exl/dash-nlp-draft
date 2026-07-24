# Shared BART-MNLI pipeline (see sic.py) - avoid loading a second ~1.6GB copy.
from sic import model

HYPOTHESIS_TEMPLATE = "This article is primarily about {}."

# ------------------------------------------------------------
# Political-relevance gate
# ------------------------------------------------------------
# Same two-way-contrast pattern as sic.py's industry-relevance gate: forcing
# every article through a 5-way orientation softmax (or a 3-way salience
# vote) meant an apolitical earnings/tech/sports article still had to "win"
# one of those labels, and BART-MNLI reliably picked a partisan-sounding one
# or "highly focused on politics" over the honest answer (not political at
# all). Deciding relevance first and defaulting non-political articles to
# Centrist/Low fixes that at the source instead of trying to out-word the
# 5-way and 3-way prompts.
_RELEVANCE_LABELS = {
    "political": "a political issue, politician, election, legislation, or government action",
    "general": (
        "general news such as business, technology, sports, entertainment, or "
        "human interest, not centered on politics or government"
    ),
}
RELEVANCE_THRESHOLD = 0.55  # min P(political) to bother with orientation/salience at all


def is_political(text, classifier=None, threshold=RELEVANCE_THRESHOLD):
    classifier = classifier or model
    result = classifier(
        text,
        candidate_labels=list(_RELEVANCE_LABELS.values()),
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    scores = dict(zip(result["labels"], result["scores"]))
    political_prob = float(scores[_RELEVANCE_LABELS["political"]])
    return {"related": political_prob >= threshold, "score": political_prob}


# ------------------------------------------------------------
# Orientation
# ------------------------------------------------------------
# "Centrist or politically neutral" is no longer a competing label here - the
# relevance gate above is what assigns Centrist now, so this list only needs
# to rank the ways an article CAN lean once it's already confirmed political.
ORIENTATION_LABELS = [
    "Progressive or left-wing",
    "Center-left",
    "Center-right",
    "Right-wing or conservative",
]


def classify_poliical_orientation(article):
    """
    classifies orientation of article using above selected model / topics
    returns dict of orienattion / ranks, and confidence levels
    """
    classification_text = article['text'][:1000]

    gate = is_political(classification_text)
    if not gate["related"]:
        return [{
            "orientation": "Centrist or politically neutral",
            "confidence": 1 - gate["score"],
            "rank": 1,
        }]

    result = model(
        classification_text,
        candidate_labels=ORIENTATION_LABELS,
        hypothesis_template="The political orientation of this news is {}",
        multi_label=False
    )

    orientation = []

    for rank in range(len(result['labels'])):
        orientation.append({
            'orientation': result['labels'][rank],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return orientation


# ------------------------------------------------------------
# Salience
# ------------------------------------------------------------
# Same reasoning as orientation: Low is now the gate's job. Down to a clean
# High-vs-Medium contest, which is what these two were actually confusable
# on (the old 3-way vote almost never picked Medium at all).
_SALIENCE_HYPOTHESES = {
    "High": "highly focused on a political issue, politician, election, or government action",
    "Medium": "touches on politics or government, but that is not the article's main focus",
}
SALIENCE_LABELS = list(_SALIENCE_HYPOTHESES.values())
SALIENCE_MAP = {v: k for k, v in _SALIENCE_HYPOTHESES.items()}


def classify_article_salience(article):
    """
    classifies topic of article using above selected model / topics
    returns dict of topics / ranks, and confidence levels
    """
    classification_text = article['text'][:1000]

    gate = is_political(classification_text)
    if not gate["related"]:
        return [{"salience": "Low", "confidence": 1 - gate["score"], "rank": 1}]

    result = model(
        classification_text,
        candidate_labels=SALIENCE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False
    )

    salience = []

    for rank in range(len(result['labels'])):
        salience.append({
            'salience': SALIENCE_MAP[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return salience
