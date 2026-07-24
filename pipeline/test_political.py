"""
Plain-assert check for the political-relevance gate in political.py.
Run: python test_political.py
"""

import political


def _stub_classifier(scores_by_label):
    """Fake zero-shot pipeline: sorts labels/scores by score desc, like the real one."""
    def classifier(text, candidate_labels, hypothesis_template, multi_label):
        pairs = sorted(
            ((label, scores_by_label[label]) for label in candidate_labels),
            key=lambda p: p[1],
            reverse=True,
        )
        return {"labels": [p[0] for p in pairs], "scores": [p[1] for p in pairs]}
    return classifier


POLITICAL = political._RELEVANCE_LABELS["political"]
GENERAL = political._RELEVANCE_LABELS["general"]

# Clearly political text -> gate lets it through.
gate = political.is_political("t", classifier=_stub_classifier({POLITICAL: 0.9, GENERAL: 0.1}))
assert gate == {"related": True, "score": 0.9}

# Clearly apolitical text -> gate blocks it.
gate = political.is_political("t", classifier=_stub_classifier({POLITICAL: 0.1, GENERAL: 0.9}))
assert gate == {"related": False, "score": 0.1}

# classify_poliical_orientation / classify_article_salience read the module-level
# `model` at call time (not a def-time default), so patching it here takes effect.
orig_model = political.model

political.model = _stub_classifier({POLITICAL: 0.1, GENERAL: 0.9})
assert political.classify_poliical_orientation({"text": "quarterly earnings call"}) == [
    {"orientation": "Centrist or politically neutral", "confidence": 0.9, "rank": 1}
]
assert political.classify_article_salience({"text": "quarterly earnings call"}) == [
    {"salience": "Low", "confidence": 0.9, "rank": 1}
]

political.model = orig_model

print("test_political: all checks passed")
