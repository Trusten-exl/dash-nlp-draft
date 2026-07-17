"""
Plain-assert check for the Wikipedia title-verification fallback in
entity_roles.py. Run: python test_entity_roles.py
"""

import entity_roles as er
from entity_roles import (
    _verified_title,
    _classify_entity,
    LABEL_MAPS,
    PERSON_LABELS,
    ORG_LABELS,
    EVENT_LABELS,
    WORK_OF_ART_LABELS,
)

RESULTS = [
    {"title": "Cristiano Ronaldo", "snippet": "..."},
    {"title": "Ronaldinho", "snippet": "..."},
]

# Model honestly says no candidate matches -> honored as None, not a fallback.
assert _verified_title(None, RESULTS) is None

# Model picks a real candidate verbatim -> returned as-is.
assert _verified_title("Ronaldinho", RESULTS) == "Ronaldinho"

# Model hallucinates a title not in the candidate list -> fall back to top hit.
assert _verified_title("Ronaldo Nazário", RESULTS) == "Cristiano Ronaldo"


def _stub_classifier(winning_description):
    """Fake zero-shot pipeline: always ranks `winning_description` first."""
    def classifier(context, candidate_labels, hypothesis_template, multi_label):
        others = [d for d in candidate_labels if d != winning_description]
        return {"labels": [winning_description] + others, "scores": [0.91] + [0.01] * len(others)}
    return classifier


def test_label_maps_cover_expected_ner_labels():
    assert set(LABEL_MAPS) == {"PERSON", "ORG", "EVENT", "WORK_OF_ART"}
    assert LABEL_MAPS["PERSON"] is PERSON_LABELS
    assert LABEL_MAPS["ORG"] is ORG_LABELS
    assert LABEL_MAPS["EVENT"] is EVENT_LABELS
    assert LABEL_MAPS["WORK_OF_ART"] is WORK_OF_ART_LABELS


def test_classify_entity_resolves_multiway_label_map():
    key, conf = _classify_entity("ctx", PERSON_LABELS, _stub_classifier(PERSON_LABELS["musician"]))
    assert key == "musician"
    assert conf == 0.91

    key, conf = _classify_entity("ctx", ORG_LABELS, _stub_classifier(ORG_LABELS["sports_team"]))
    assert key == "sports_team"

    key, conf = _classify_entity("ctx", WORK_OF_ART_LABELS, _stub_classifier(WORK_OF_ART_LABELS["movie_or_tv_show"]))
    assert key == "movie_or_tv_show"


def test_classify_entity_roles_caps_enrichment_by_mention_count():
    # 25 PERSON entities that all classify as "athlete", plus one "other"
    # entity with a huge mention_count that must never get enriched.
    role_by_text = {f"Player{i}": "athlete" for i in range(25)}
    role_by_text["Extra"] = "other"

    def stub_classifier(context, candidate_labels, hypothesis_template, multi_label):
        role = role_by_text.get(context, "other")
        winning_desc = PERSON_LABELS[role]
        others = [d for d in candidate_labels if d != winning_desc]
        return {"labels": [winning_desc] + others, "scores": [0.91] + [0.01] * len(others)}

    entities = [
        {"entity_text": f"Player{i}", "entity_label": "PERSON", "mention_count": i + 1}
        for i in range(25)
    ]
    entities.append({"entity_text": "Extra", "entity_label": "PERSON", "mention_count": 999})

    enriched_calls = []

    def stub_get_celebrity_info(context, name):
        enriched_calls.append(name)
        return {"url": f"https://en.wikipedia.org/wiki/{name}"}

    original = er.get_celebrity_info
    er.get_celebrity_info = stub_get_celebrity_info
    try:
        roles = er.classify_entity_roles(
            {"text": ""}, entities, classifier=stub_classifier, enrich_cap=20
        )
    finally:
        er.get_celebrity_info = original

    enriched_names = {r["entity_text"] for r in roles if r["url"] is not None}
    expected_enriched = {f"Player{i}" for i in range(5, 25)}  # top 20 of 25 by mention_count
    assert enriched_names == expected_enriched
    assert len(enriched_names) == 20
    assert "Extra" not in enriched_names  # role == "other" never enriched, regardless of count
    assert len(enriched_calls) == 20

    # Every input entity still comes back out, whether enriched or not.
    assert len(roles) == 26
    extra_row = next(r for r in roles if r["entity_text"] == "Extra")
    assert extra_row["role"] == "other"
    assert extra_row["url"] is None


test_label_maps_cover_expected_ner_labels()
test_classify_entity_resolves_multiway_label_map()
test_classify_entity_roles_caps_enrichment_by_mention_count()
print("ok")
