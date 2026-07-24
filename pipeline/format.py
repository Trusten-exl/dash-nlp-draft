# Concise, discriminative phrases. The previous paragraph-length descriptions
# overlapped so much that BART-MNLI's softmax picked "Article" (the generic
# catch-all) for 89/89 GT articles regardless of the true format - the same
# lesson sic.py already learned for SIC divisions (see its comment on
# DIVISION_HYPOTHESES): short, concrete phrases classify far better than long
# overlapping paragraphs.
MAPPING = {
    "Breaking": "breaking news about a very recent event, updated as facts come in",

    "Feature": "an in-depth feature or narrative profile with storytelling and human-interest detail",

    "Analysis": "analysis of the causes, significance, or implications of events, not just what happened",

    "Game Recap": "a recap of a sports game's final score, key plays, and standout performances",

    "Off-Field News": "sports business news such as trades, contracts, injuries, or league policy, not describing a game itself",

    "Article": "a routine, straightforward news article reporting facts without urgency, narrative, or analysis",
}

FORMATS = list(MAPPING.values())
MAPPED = {v: k for k, v in MAPPING.items()}

# Shared BART-MNLI pipeline (see sic.py) - avoid loading a second ~1.6GB copy.
from sic import model

def classify_article_format(article):
    """
    classifies format of article using above selected model / format
    returns dict of format / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=FORMATS,
        hypothesis_template="This article is {}",
        multi_label=False
    )

    # sort into format list for use in sql
    format = []

    for rank in range(len(result['labels'])):
        format.append({
            'format': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return format
