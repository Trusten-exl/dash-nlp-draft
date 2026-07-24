# Concise, discriminative phrases. The previous paragraph-length descriptions
# overlapped so much that BART-MNLI's softmax picked "Report" (the generic
# catch-all) for 89/89 GT articles regardless of the true intent - the same
# lesson sic.py already learned for SIC divisions (see its comment on
# DIVISION_HYPOTHESES): short, concrete phrases classify far better than long
# overlapping paragraphs.
MAPPING = {
    "Report": "a factual report of recent events, focused on what happened rather than opinion or interpretation",

    "Opinion": "an opinion piece arguing for a viewpoint, judgment, or recommendation",

    "Explainer": "an explainer that provides background and context to help readers understand a topic",

    "Analysis": "an analysis of the causes, significance, or broader implications of events",

    "Guide": "a how-to guide with practical, step-by-step instructions or advice",

    "Review": "a review evaluating a product, service, or creative work",
}

ARTICLE_INTENTS = list(MAPPING.values())
MAPPED = {v: k for k, v in MAPPING.items()}

# Shared BART-MNLI pipeline (see sic.py) - avoid loading a second ~1.6GB copy.
from sic import model

def classify_article_intent(article):
    """
    classifies intent of article using above selected model / intent
    returns dict of intent / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=ARTICLE_INTENTS,
        hypothesis_template="This article is {}",
        multi_label=False
    )

    # sort into intents list for use in sql
    intents = []

    for rank in range(len(result['labels'])):
        intents.append({
            'intent': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return intents
