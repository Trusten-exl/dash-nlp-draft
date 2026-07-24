"""
Plain-assert check for the two-stage gate in readability.py's
_classify_spectrum. Run: python test_readability.py
"""

import readability as rd


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

# Low side wins the gate, and within it the second (higher) level wins.
gate_low, gate_high = rd.AUDIENCE_GATE
split_lo, split_hi = rd.AUDIENCE_LOW_SPLIT
result = rd._classify_spectrum(
    rd.AUDIENCE_LEVELS, rd.AUDIENCE_GATE, rd.AUDIENCE_LOW_SPLIT, rd.AUDIENCE_HIGH_SPLIT,
    "text", _stub_classifier({gate_low: 0.8, gate_high: 0.2, split_lo: 0.3, split_hi: 0.7}),
)
assert result["label"] == "Informed Reader", result
assert 0 < result["score"] < 50, result  # low half of the spectrum
assert abs(result["confidence"] - 0.8 * 0.7) < 1e-9, result

# High side wins the gate, and within it the first (lower of the two) level wins.
gate_low, gate_high = rd.MATURITY_GATE
split_lo, split_hi = rd.MATURITY_HIGH_SPLIT
result = rd._classify_spectrum(
    rd.MATURITY_LEVELS, rd.MATURITY_GATE, rd.MATURITY_LOW_SPLIT, rd.MATURITY_HIGH_SPLIT,
    "text", _stub_classifier({gate_low: 0.1, gate_high: 0.9, split_lo: 0.6, split_hi: 0.4}),
)
assert result["label"] == "Sophisticated", result
assert 50 < result["score"] < 100, result  # high half of the spectrum
assert abs(result["confidence"] - 0.9 * 0.6) < 1e-9, result

print("test_readability: all checks passed")
