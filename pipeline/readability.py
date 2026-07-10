# ============================================================
# readability.py — intended audience & writing maturity via NLP
# ============================================================
# Replaces the render-time Flesch-Kincaid heuristics with zero-shot
# classification. Reuses the shared BART-MNLI pipeline already loaded in
# sic.py so we don't hold a second ~1.6GB model in memory.

from sic import model


# Ordered low → high. Each level pairs a stored label with the natural-language
# description fed to the zero-shot model.
AUDIENCE_LEVELS = [
    ("General Public",      "written for a general audience with no specialized background"),
    ("Informed Reader",     "written for an informed reader who regularly follows the news"),
    ("Professional",        "written for professionals working in the relevant field"),
    ("Expert / Specialist", "written for experts or specialists with deep domain knowledge"),
]

MATURITY_LEVELS = [
    ("Accessible",    "written in simple, plain, easy-to-read language"),
    ("Standard",      "written in clear, standard, everyday prose"),
    ("Sophisticated", "written in sophisticated, nuanced, complex language"),
    ("Advanced",      "written in dense, advanced, highly technical language"),
]

HYPOTHESIS_TEMPLATE = "This article is {}."


def _article_text(article, max_chars=1500):
    """Combine the most informative fields for classification."""
    parts = [
        article.get("title") or "",
        article.get("description") or "",
        article.get("text") or "",
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())[:max_chars]


def _classify_spectrum(levels, text, classifier):
    """
    Zero-shot over ordered levels → a smooth 0–100 spectrum position plus the
    single most likely label.

    The score is the probability-weighted midpoint across the levels, so an
    article that is split between (say) "Professional" and "Expert" lands
    between their bands rather than snapping to one.
    """
    descriptions = [desc for _, desc in levels]

    result = classifier(
        text,
        candidate_labels=descriptions,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    prob = dict(zip(result["labels"], result["scores"]))

    n = len(levels)
    score = sum(
        prob[desc] * ((i + 0.5) / n * 100)
        for i, (_, desc) in enumerate(levels)
    )

    top_desc = result["labels"][0]
    top_label = next(name for name, desc in levels if desc == top_desc)

    return {
        "score": float(score),
        "label": top_label,
        "confidence": float(result["scores"][0]),
    }


def classify_readability(article, classifier=model):
    """
    Classify intended audience and writing maturity via zero-shot NLP.

    Returns
    -------
    dict
        {
          "audience": {"score": float, "label": str, "confidence": float},
          "maturity": {"score": float, "label": str, "confidence": float},
        }
    """
    text = _article_text(article)

    if not text:
        return {
            "audience": {"score": 50.0, "label": "Informed Reader", "confidence": 0.0},
            "maturity": {"score": 50.0, "label": "Standard", "confidence": 0.0},
        }

    return {
        "audience": _classify_spectrum(AUDIENCE_LEVELS, text, classifier),
        "maturity": _classify_spectrum(MATURITY_LEVELS, text, classifier),
    }
