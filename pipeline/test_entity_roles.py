"""
Plain-assert check for the Wikipedia title-verification fallback in
entity_roles.py. Run: python test_entity_roles.py
"""

from entity_roles import _verified_title

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

print("ok")
