# ============================================================
# readability.py — intended audience & writing maturity via NLP
# ============================================================
# Replaces the render-time Flesch-Kincaid heuristics with zero-shot
# classification. Reuses the shared BART-MNLI pipeline already loaded in
# sic.py so we don't hold a second ~1.6GB model in memory.

from sic import model

HYPOTHESIS_TEMPLATE = "This article is {}."

# Ordered low → high, purely for labeling/scoring - the actual classification
# is two binary gates (see *_GATE / *_SPLIT below), not a flat vote across
# all four. A flat 4-way softmax kept collapsing to whichever description
# sounds like a safe middle-of-the-road default: on the GT set, "Informed
# Reader" won 67/89 articles and "Sophisticated" won 79/89, regardless of the
# true label. Splitting into a general-vs-specialized (or plain-vs-complex)
# gate, then a second contrast on whichever side wins, gives BART-MNLI an
# actual two-way call to make instead of a 4-way vote where nothing stands
# out - the same fix already applied to political orientation/salience.
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

AUDIENCE_GATE = (
    "written for a general audience with no specialized background",
    "written assuming specialized professional or expert knowledge",
)
AUDIENCE_LOW_SPLIT = (
    "written assuming no prior familiarity with the topic",
    "written assuming the reader already follows related news",
)
AUDIENCE_HIGH_SPLIT = (
    "written for professionals or practitioners in the field",
    "written for deep subject-matter experts, using technical or academic language",
)

MATURITY_GATE = (
    "written in simple, plain, everyday language",
    "written in complex, dense, or technical language",
)
MATURITY_LOW_SPLIT = (
    "written in very simple, casual, conversational language",
    "written in clear, professional prose typical of mainstream journalism",
)
MATURITY_HIGH_SPLIT = (
    "sophisticated but still readable by an educated general audience",
    "dense, highly technical, jargon-heavy language requiring specialized expertise",
)


def _article_text(article, max_chars=1500):
    """Combine the most informative fields for classification."""
    parts = [
        article.get("title") or "",
        article.get("description") or "",
        article.get("text") or "",
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())[:max_chars]


def _classify_spectrum(levels, gate, low_split, high_split, text, classifier):
    """
    Two-stage zero-shot over four ordered levels → a smooth 0-100 spectrum
    position plus the single most likely label.

    Stage 1 is a binary gate (low half vs. high half of `levels`). Stage 2
    only runs the split for whichever half won, contrasting its two levels;
    the losing half's two levels split its remaining probability mass evenly
    (they don't affect the label, only nudge the continuous score).
    """
    gate_low, gate_high = gate

    top = classifier(text, candidate_labels=[gate_low, gate_high],
                      hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False)
    p_low = dict(zip(top["labels"], top["scores"]))[gate_low]

    if p_low >= 0.5:
        sub = classifier(text, candidate_labels=list(low_split),
                          hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False)
        sub_scores = dict(zip(sub["labels"], sub["scores"]))
        probs = [
            p_low * sub_scores[low_split[0]],
            p_low * sub_scores[low_split[1]],
            (1 - p_low) * 0.5,
            (1 - p_low) * 0.5,
        ]
    else:
        sub = classifier(text, candidate_labels=list(high_split),
                          hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=False)
        sub_scores = dict(zip(sub["labels"], sub["scores"]))
        probs = [
            p_low * 0.5,
            p_low * 0.5,
            (1 - p_low) * sub_scores[high_split[0]],
            (1 - p_low) * sub_scores[high_split[1]],
        ]

    n = len(levels)
    score = sum(p * ((i + 0.5) / n * 100) for i, p in enumerate(probs))
    best_i = max(range(n), key=lambda i: probs[i])

    return {
        "score": float(score),
        "label": levels[best_i][0],
        "confidence": float(probs[best_i]),
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
        "audience": _classify_spectrum(
            AUDIENCE_LEVELS, AUDIENCE_GATE, AUDIENCE_LOW_SPLIT, AUDIENCE_HIGH_SPLIT, text, classifier
        ),
        "maturity": _classify_spectrum(
            MATURITY_LEVELS, MATURITY_GATE, MATURITY_LOW_SPLIT, MATURITY_HIGH_SPLIT, text, classifier
        ),
    }
