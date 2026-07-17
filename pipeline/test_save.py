"""
Plain-assert check for the alias-merge logic in save.py.
Run: cd pipeline && python test_save.py
"""

from save import _merge_duplicate_urls

SINNER_URL = "https://en.wikipedia.org/wiki/Jannik_Sinner"
DJOKOVIC_URL = "https://en.wikipedia.org/wiki/Novak_Djokovic"

roles = [
    {"entity_text": "Sinner", "entity_label": "PERSON", "role": "athlete",
     "url": SINNER_URL, "confidence": 0.8, "mention_count": 3},
    {"entity_text": "Jannik Sinner", "entity_label": "PERSON", "role": "athlete",
     "url": SINNER_URL, "confidence": 0.9, "mention_count": 5},
    {"entity_text": "Novak Djokovic", "entity_label": "PERSON", "role": "athlete",
     "url": DJOKOVIC_URL, "confidence": 0.95, "mention_count": 2},
    {"entity_text": "Random Coach", "entity_label": "PERSON", "role": "other",
     "url": None, "confidence": 0.0, "mention_count": 1},
]

merged = _merge_duplicate_urls(roles)

by_url = {r["url"]: r for r in merged if r["url"]}

sinner = by_url[SINNER_URL]
assert sinner["entity_text"] == "Jannik Sinner"   # longer name wins as canonical
assert sinner["mention_count"] == 8               # 3 + 5, summed rather than dropped

djokovic = by_url[DJOKOVIC_URL]
assert djokovic["mention_count"] == 2             # untouched, no duplicate to merge

no_url = [r for r in merged if r["url"] is None]
assert len(no_url) == 1
assert no_url[0]["entity_text"] == "Random Coach"  # entities without a url pass through unmerged

assert len(merged) == 3  # 2 Sinner rows collapsed into 1, plus Djokovic + Random Coach

print("ok")
