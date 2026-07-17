# Entity Taxonomy Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify entities into a broader taxonomy (athlete, actor, musician,
politician, sports team, sporting event, movie/TV show) across every article
(not just Sports), with Wikipedia enrichment cost-capped per article, alias
mentions correctly merged and summed, and the dashboard showing this via an
extended Sports Highlights card (Teams section) plus a new full-width
"Important Entities" widget for every other article type.

**Architecture:** `entity_roles.py` gains a multi-way label map per NER type
(`LABEL_MAPS`) and a two-tier classify/enrich pipeline
(`classify_entity_roles`): cheap local zero-shot role classification runs on
every matched entity, but the expensive Claude+Wikipedia enrichment
(`get_celebrity_info`) only runs on the top `ENRICH_CAP` entities per article
by mention count. `save.py` gains a pure `_merge_duplicate_urls` helper that
sums mention counts across aliases resolving to the same Wikipedia page
(fixing an existing bug where the losing alias's count was silently
dropped). `reclassify_entities.py` drops its Sports-only filter and widens
its entity query. The dashboard reads directly from the already-merged
`entity_roles` table instead of joining raw `entities` + an allow-list.

**Tech Stack:** Python, spaCy (NER labels), local BART-MNLI zero-shot
classifier (`sic.model`), Claude API (`anthropic` SDK, already wired in
`entity_roles.py`), SQLite (`pipeline/db/connection.py`), Streamlit dashboard.

## Global Constraints

- Follow this repo's existing test convention: plain Python scripts with
  `assert` statements and a trailing `print("ok")`, run via
  `cd pipeline && python test_X.py` — no pytest anywhere in this codebase.
- `ENRICH_CAP = 20` (max entities per article that get Wikipedia/Claude
  enrichment) — a module-level constant in `entity_roles.py`, easy to tune
  later.
- All new/modified code in `pipeline/` must remain importable without
  `torch`/`transformers` actually running inference — tests stub the
  `classifier` and `get_celebrity_info` callables rather than invoking real
  models or the network (matching how `pipeline/test_entity_roles.py`
  already stubs `sic` to avoid loading the real BART model at import time).
- DB schema changes must be additive and migration-safe: use
  `ALTER TABLE ... ADD COLUMN` guarded by a `PRAGMA table_info` check, since
  `CREATE TABLE IF NOT EXISTS` does not add columns to an already-existing
  table (the user's dev DB already has an `entity_roles` table without the
  new `mention_count` column).

---

### Task 1: Expand the classification taxonomy

**Files:**
- Modify: `pipeline/ner.py:5-12`
- Modify: `pipeline/entity_roles.py:81-98`
- Test: `pipeline/test_entity_roles.py` (append)

**Interfaces:**
- Produces: `entity_roles.PERSON_LABELS` (dict, now 5 keys: athlete, actor,
  musician, politician, other), `entity_roles.ORG_LABELS` (dict: sports_team,
  other), `entity_roles.EVENT_LABELS` (unchanged: sporting_event, other),
  `entity_roles.WORK_OF_ART_LABELS` (dict: movie_or_tv_show, other),
  `entity_roles.LABEL_MAPS` (dict: `{"PERSON": PERSON_LABELS, "ORG":
  ORG_LABELS, "EVENT": EVENT_LABELS, "WORK_OF_ART": WORK_OF_ART_LABELS}`).
  Task 2 consumes `LABEL_MAPS` directly.

- [ ] **Step 1: Write the failing test**

Append to `pipeline/test_entity_roles.py` (after the existing
`_verified_title` asserts, before the final `print("ok")` — remove the
existing `print("ok")` line since more asserts follow it now):

```python
from entity_roles import (
    LABEL_MAPS,
    PERSON_LABELS,
    ORG_LABELS,
    EVENT_LABELS,
    WORK_OF_ART_LABELS,
    _classify_entity,
)


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


test_label_maps_cover_expected_ner_labels()
test_classify_entity_resolves_multiway_label_map()
print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python test_entity_roles.py`
Expected: `ImportError: cannot import name 'LABEL_MAPS' from 'entity_roles'`
(or similar — `ORG_LABELS`/`WORK_OF_ART_LABELS` don't exist yet either).

- [ ] **Step 3: Implement the taxonomy**

In `pipeline/entity_roles.py`, replace lines 81-98 (from `HYPOTHESIS_TEMPLATE
= ...` through the `CONFIDENCE = 0.55` line) with:

```python
HYPOTHESIS_TEMPLATE = "This text is about {}."

# Each entry maps a zero-shot description to the role key it represents.
# _classify_entity picks whichever description the classifier ranks highest
# across the whole set (not a binary positive-vs-other pair) and maps it
# back to its key; "other" is just one more candidate in the set.
PERSON_LABELS = {
    "athlete": "a professional athlete, sports player, or competitor",
    "actor": "a professional actor or actress",
    "musician": "a musician or singer",
    "politician": "a politician or government official",
    "other": "a person who is none of the above",
}

ORG_LABELS = {
    "sports_team": "a professional or amateur sports team or club",
    "other": "an organization that is not a sports team",
}

EVENT_LABELS = {
    "sporting_event": "a sports competition, game, match, tournament, or championship",
    "other": "an event that is not a sporting event",
}

WORK_OF_ART_LABELS = {
    "movie_or_tv_show": "a movie, film, or television show",
    "other": "a creative work that is not a movie or TV show",
}

LABEL_MAPS = {
    "PERSON": PERSON_LABELS,
    "ORG": ORG_LABELS,
    "EVENT": EVENT_LABELS,
    "WORK_OF_ART": WORK_OF_ART_LABELS,
}

# Minimum probability for the winning label before we accept it. Tuned
# conservatively so displayed entities err toward precision over recall.
# Adjust after eyeballing reclassify output.
CONFIDENCE = 0.55
```

In `pipeline/ner.py`, replace lines 5-12 (`KEEP_LABELS = {...}`) with:

```python
KEEP_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "EVENT",
    "WORK_OF_ART",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python test_entity_roles.py`
Expected: prints `ok` with no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add pipeline/entity_roles.py pipeline/ner.py pipeline/test_entity_roles.py
git commit -m "Expand entity role taxonomy beyond sports (actors, musicians, politicians, teams, movies/TV)"
```

---

### Task 2: Two-tier classify (all entities) + enrich (capped top-N)

**Files:**
- Modify: `pipeline/entity_roles.py:200-253` (replace `classify_sports_entities`)
- Test: `pipeline/test_entity_roles.py` (append)

**Interfaces:**
- Consumes: `entity_roles.LABEL_MAPS`, `entity_roles.CONFIDENCE`,
  `entity_roles._entity_context(text, entity_text)`,
  `entity_roles._classify_entity(context, label_map, classifier)`,
  `entity_roles.get_celebrity_info(context, name)` (all from Task 1 / existing code).
- Produces: `entity_roles.classify_entity_roles(article, entities,
  classifier=model, enrich_cap=ENRICH_CAP)` returning a list of dicts with
  keys `entity_text, entity_label, role, url, confidence, mention_count`.
  Task 4 (`reclassify_entities.py`) calls this by name. Replaces
  `classify_sports_entities` (deleted — no other call sites besides
  `reclassify_entities.py`, updated in Task 4).

- [ ] **Step 1: Write the failing test**

Append to `pipeline/test_entity_roles.py` (before the final `test_label_maps...`
calls and `print("ok")` — move `print("ok")` to the very end again):

```python
import entity_roles as er


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


test_classify_entity_roles_caps_enrichment_by_mention_count()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python test_entity_roles.py`
Expected: `AttributeError: module 'entity_roles' has no attribute
'classify_entity_roles'`

- [ ] **Step 3: Implement the two-tier classify/enrich function**

In `pipeline/entity_roles.py`, replace lines 200-253 (the whole
`classify_sports_entities` function) with:

```python
ENRICH_CAP = 20  # max entities per article that get Wikipedia/Claude enrichment


def classify_entity_roles(article, entities, classifier=model, enrich_cap=ENRICH_CAP):
    """
    Classify each recognized entity (PERSON/ORG/EVENT/WORK_OF_ART) into a
    specific role via zero-shot MNLI, then enrich with a Wikipedia link only
    for the top `enrich_cap` entities per article (by mention_count) that got
    a real role (role != "other"). Enrichment is 2 Claude + 2 Wikipedia calls
    per entity, so it's capped independently of how many entities merely get
    a cheap (local-model) role label.

    Parameters
    ----------
    article : mapping with a "text" key (the article body for context).
    entities : iterable of mappings, each with "entity_text", "entity_label",
        and "mention_count".
    classifier : the shared zero-shot pipeline (defaults to the BART-MNLI model).
    enrich_cap : max number of non-"other" entities to enrich per article.

    Returns
    -------
    list of dicts: {entity_text, entity_label, role, url, confidence, mention_count}
    """
    text = article.get("text") or ""
    classified = []

    for ent in entities:
        etext = ent["entity_text"]
        elabel = ent["entity_label"]
        mention_count = ent.get("mention_count") or 1
        label_map = LABEL_MAPS.get(elabel)

        if label_map is not None:
            context = _entity_context(text, etext)
            key, conf = _classify_entity(context, label_map, classifier)
            role = key if conf >= CONFIDENCE else "other"
        else:
            context, role, conf = etext, "other", 0.0

        classified.append({
            "entity_text": etext,
            "entity_label": elabel,
            "role": role,
            "confidence": conf,
            "mention_count": mention_count,
            "context": context,
        })

    eligible = [c for c in classified if c["role"] != "other"]
    eligible.sort(key=lambda c: c["mention_count"], reverse=True)
    for c in eligible[:enrich_cap]:
        c["_enrich"] = True

    out = []
    for c in classified:
        if c.get("_enrich"):
            info = get_celebrity_info(c["context"], c["entity_text"])
            url = info.get("url") if info else None
        else:
            url = None
        out.append({
            "entity_text": c["entity_text"],
            "entity_label": c["entity_label"],
            "role": c["role"],
            "url": url,
            "confidence": c["confidence"],
            "mention_count": c["mention_count"],
        })

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python test_entity_roles.py`
Expected: prints `ok` with no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add pipeline/entity_roles.py pipeline/test_entity_roles.py
git commit -m "Replace classify_sports_entities with classify_entity_roles (cap Wikipedia/Claude enrichment per article)"
```

---

### Task 3: Fix alias merging to sum mention_count instead of dropping it

**Files:**
- Modify: `pipeline/save.py:371-` (the `save_entity_roles` function)
- Test: `pipeline/test_save.py` (new)

**Interfaces:**
- Produces: `save._merge_duplicate_urls(roles)` — pure function, list of role
  dicts in, list of role dicts out (merged). `save.save_entity_roles(roles,
  article_id)` now requires each `roles` dict to include a `mention_count`
  key (guaranteed by Task 2's `classify_entity_roles` output).

- [ ] **Step 1: Write the failing test**

Create `pipeline/test_save.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pipeline && python test_save.py`
Expected: `ImportError: cannot import name '_merge_duplicate_urls' from 'save'`

- [ ] **Step 3: Implement the merge helper and wire it into save_entity_roles**

In `pipeline/save.py`, replace the `save_entity_roles` function (currently
lines 371-onward, ending right before the blank line preceding
`def save_readability`) with:

```python
def _merge_duplicate_urls(roles):
    """
    Group entities by resolved Wikipedia url, summing mention_count across
    aliases (e.g. "Sinner" and "Jannik Sinner") and keeping the longer
    entity_text as the canonical display name. Entities with no url (not
    enriched, or no Wikipedia match) pass through unmerged.
    """
    merged = {}
    unresolved = []

    for r in roles:
        url = r.get("url")
        if not url:
            unresolved.append(dict(r))
            continue

        if url not in merged:
            merged[url] = dict(r)
        else:
            total_mentions = merged[url]["mention_count"] + r["mention_count"]
            canonical = r if len(r["entity_text"]) > len(merged[url]["entity_text"]) else merged[url]
            merged[url] = dict(canonical)
            merged[url]["mention_count"] = total_mentions

    return list(merged.values()) + unresolved


def save_entity_roles(roles, article_id):
    """
    sql code for saving per-entity roles (athlete / actor / musician /
    politician / sports_team / sporting_event / movie_or_tv_show / other).
    Clears any existing rows for the article first so re-runs don't
    accumulate duplicates. Aliases resolving to the same Wikipedia url are
    merged (see _merge_duplicate_urls) so mention counts aren't lost.
    """
    execute("DELETE FROM entity_roles WHERE article_id = ?", (article_id,))

    for r in _merge_duplicate_urls(roles):
        execute("""
        INSERT INTO entity_roles (
            article_id, entity_text, entity_label, role, url, confidence, mention_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            r['entity_text'],
            r['entity_label'],
            r['role'],
            r['url'],
            r['confidence'],
            r['mention_count'],
        ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd pipeline && python test_save.py`
Expected: prints `ok` with no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add pipeline/save.py pipeline/test_save.py
git commit -m "Sum mention_count when merging aliased entities instead of dropping it"
```

---

### Task 4: Classify all articles, widen the entity query, migrate the schema

**Files:**
- Modify: `pipeline/reclassify_entities.py` (whole file)
- Modify: `pipeline/db/create_tables.py:227-243` (the `entity_roles` table def, for consistency)
- No automated test: this task is DB/query wiring with no pure logic to
  isolate (matches this repo's existing convention — there is no
  DB-integration test anywhere in the codebase today). Verified manually in
  Task 5's final smoke test.

**Interfaces:**
- Consumes: `entity_roles.classify_entity_roles` (Task 2),
  `entity_roles.CONFIDENCE` (Task 1), `save.save_entity_roles` (Task 3).
- Produces: `reclassify_entities.classifiable_article_ids()` (replaces
  `sports_article_ids()`), `reclassify_entities.ensure_tables()` (now
  migration-safe for the new `mention_count` column).

- [ ] **Step 1: Update the schema definitions**

In `pipeline/db/create_tables.py`, replace lines 227-243 with:

```python
# per-entity role (zero-shot NLP): athlete / actor / musician / politician /
# sports_team / sporting_event / movie_or_tv_show / other
cursor.execute("""
CREATE TABLE IF NOT EXISTS entity_roles (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    article_id INTEGER,

    entity_text TEXT,

    entity_label TEXT,

    role TEXT,

    url TEXT,

    confidence REAL,

    mention_count INTEGER
)
""")
```

- [ ] **Step 2: Rewrite reclassify_entities.py**

Replace the entire contents of `pipeline/reclassify_entities.py` with:

```python
"""
Populate the entity_roles table for every article stored in the DB.

Classifies each PERSON / ORG / EVENT / WORK_OF_ART entity into a specific
role (athlete, actor, musician, politician, sports_team, sporting_event,
movie_or_tv_show, or other) via zero-shot MNLI, using the article text for
context, and writes the result to entity_roles. The dashboard reads this to
render the Sports Highlights card (sports articles) or the Important
Entities widget (every other article).

Only entities that get a real role (not "other") and rank in the top
ENRICH_CAP by mention_count get a Wikipedia link (see
entity_roles.classify_entity_roles) — this keeps the number of Claude/
Wikipedia calls bounded regardless of how many articles or entities exist.

Run from the pipeline/ directory on a machine where torch/transformers work:

    cd pipeline
    python reclassify_entities.py
"""

from db.connection import query, get_conn, execute
from entity_roles import classify_entity_roles, CONFIDENCE
from save import save_entity_roles

ENTITY_LABELS = ("PERSON", "ORG", "EVENT", "WORK_OF_ART")


def clear_roles():
    execute("DELETE FROM sqlite_sequence WHERE name='entity_roles'")
    execute("DELETE FROM entity_roles")


def ensure_tables():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            entity_text TEXT,
            entity_label TEXT,
            role TEXT,
            url TEXT,
            confidence REAL,
            mention_count INTEGER
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(entity_roles)")}
    if "mention_count" not in existing_columns:
        conn.execute("ALTER TABLE entity_roles ADD COLUMN mention_count INTEGER")
    conn.commit()
    conn.close()


def classifiable_article_ids():
    """Every article_id in the DB."""
    df = query("SELECT article_id FROM articles ORDER BY article_id")
    return [int(a) for a in df["article_id"]]


def main():
    ensure_tables()

    ids = classifiable_article_ids()
    total = len(ids)
    print(f"Classifying entities for {total} articles "
          f"(positive-label threshold={CONFIDENCE})...\n")

    placeholders = ", ".join("?" for _ in ENTITY_LABELS)

    for i, aid in enumerate(ids):
        art = query("SELECT text FROM articles WHERE article_id = ?", (aid,))
        if art.empty:
            continue
        article = {"text": art.iloc[0]["text"]}

        entities = query(
            f"SELECT entity_text, entity_label, mention_count FROM entities "
            f"WHERE article_id = ? AND entity_label IN ({placeholders})",
            (aid, *ENTITY_LABELS),
        ).to_dict("records")

        roles = classify_entity_roles(article, entities)
        save_entity_roles(roles, aid)

        matched = [r["entity_text"] for r in roles if r["role"] != "other"]
        print(f"[{i + 1}/{total}] id={aid}  matched={matched or '—'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/reclassify_entities.py pipeline/db/create_tables.py
git commit -m "Classify entity roles for all articles, not just Sports; add mention_count migration"
```

---

### Task 5: Dashboard — Teams section + full-width Important Entities widget

**Files:**
- Modify: `dash_pages/10_Article_Detail.py:472-544` (delete `_allowed_entities`
  + `_entity_chips`, replace `sports_highlights_html`, add `_role_chips` and
  `important_entities_html`)
- Modify: `dash_pages/10_Article_Detail.py:893` (call site: drop the `entities` arg)
- Modify: `dash_pages/10_Article_Detail.py` (insert new full-width section
  after the bottom_right column block, before the "Analysis Details" expander)
- No automated test: Streamlit UI rendering, no existing UI test harness in
  this repo. Verified manually below.

**Interfaces:**
- Consumes: `entity_roles` DataFrame as loaded by the existing
  `load_roles(article_id)` (columns: `entity_text, entity_label, role, url,
  confidence, mention_count` — matches Task 3/4's schema), `topics` DataFrame
  and `is_sports_article(topics)` (existing, unchanged).
- Produces: `_role_chips(roles_df, role_name, accent, limit=10)`,
  `sports_highlights_html(roles)` (signature changed — no longer takes
  `entities`), `important_entities_html(roles)`.

- [ ] **Step 1: Replace the entity-chip rendering functions**

In `dash_pages/10_Article_Detail.py`, delete lines 472-521 (`_allowed_entities`
and `_entity_chips` in full) and replace lines 524-544
(`sports_highlights_html`) with:

```python
def _role_chips(roles, role_name: str, accent: str, limit: int = 10) -> str | None:
    """
    Pill list of the top `limit` entities of a given resolved role, sorted by
    (alias-merged) mention_count. Entities outside the enrichment cap have no
    url and render as a plain (non-linked) chip.
    """
    if roles is None or roles.empty:
        return None
    df = roles[roles["role"] == role_name]
    if df.empty:
        return None
    df = df.sort_values("mention_count", ascending=False).head(limit)

    chips = ""
    for _, row in df.iterrows():
        name = html.escape(str(row["entity_text"]))
        count = int(row["mention_count"]) if pd.notna(row["mention_count"]) else 1
        badge = (
            f'<span style="opacity:.55;font-size:.62rem;margin-left:5px;">{count}</span>'
            if count > 1 else ""
        )
        pill = (
            f'<span style="display:inline-block;background:{accent}18;color:{accent};'
            f'border:1px solid {accent}55;border-radius:999px;padding:3px 10px;'
            f'margin:0 6px 7px 0;font-size:.78rem;font-weight:600;">{name}{badge}</span>'
        )
        url = row["url"]
        if pd.notna(url) and url:
            chips += f'<a href="{url}" target="_blank" class="entity-link">{pill}</a>'
        else:
            chips += pill
    return chips


def _labeled_section(sub_label: str, chips: str | None) -> str:
    if not chips:
        return ""
    return (
        f'<div style="font-size:.58rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:.04rem;margin:2px 0 7px;">{sub_label}</div>'
        f'<div style="margin-bottom:10px;">{chips}</div>'
    )


def sports_highlights_html(roles) -> str | None:
    """Card listing the key athletes, teams, and major events named in the article."""
    athletes = _role_chips(roles, "athlete", "#1f77b4")
    teams = _role_chips(roles, "sports_team", "#2ca02c")
    events = _role_chips(roles, "sporting_event", "#d62728")
    if athletes is None and teams is None and events is None:
        return None

    body = (
        _labeled_section("Athletes - Top 10", athletes)
        + _labeled_section("Teams", teams)
        + _labeled_section("Major Events", events)
    )
    return (
        '<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);'
        f'padding:12px 14px;background:rgba(0,0,0,.02);">{body}</div>'
    )


def important_entities_html(roles) -> str | None:
    """
    Full-width card of notable entities for non-sports articles: actors,
    musicians, politicians, movies/TV shows, and organizations/teams,
    grouped by category and sorted by (alias-merged) mention count.
    """
    sections = [
        ("Actors", "actor", "#9467bd"),
        ("Musicians", "musician", "#e377c2"),
        ("Politicians", "politician", "#8c564b"),
        ("Movies / TV Shows", "movie_or_tv_show", "#17becf"),
        ("Organizations", "sports_team", "#2ca02c"),
    ]

    rendered = "".join(
        _labeled_section(label, _role_chips(roles, role_name, accent, limit=15))
        for label, role_name, accent in sections
    )
    if not rendered:
        return None

    return (
        '<div style="border-radius:12px;border:1px solid rgba(120,120,120,.25);'
        f'padding:14px 16px;background:rgba(0,0,0,.02);">{rendered}</div>'
    )
```

- [ ] **Step 2: Update the Sports Highlights call site**

Change (originally around line 893, now shifted — search for the line):
```python
                sports_html = sports_highlights_html(entities, roles)
```
to:
```python
                sports_html = sports_highlights_html(roles)
```

- [ ] **Step 3: Add the full-width Important Entities section**

Find the end of the `bottom_left, bottom_right = st.columns(2, gap="large")`
block — the `salience_badge_html(...)` line is the last line of that block,
immediately followed by a blank line and then the `# Analysis Details
(expander)` comment block. Insert this new block between them, at the same
indentation level as `top_left, top_right = st.columns(...)` /
`bottom_left, bottom_right = st.columns(...)` (8 spaces):

```python
        if not is_sports_article(topics):
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.markdown(section_label_html("Important Entities"), unsafe_allow_html=True)
            important_html = important_entities_html(roles)
            if important_html is not None:
                st.markdown(important_html, unsafe_allow_html=True)
            else:
                st.info("No notable entities detected.")
```

- [ ] **Step 4: Manual smoke test**

This task has no automated test (Streamlit UI, no existing UI test harness
in this repo — matches the codebase's convention). Verify by hand once
Task 4's pipeline has been run:

```bash
cd pipeline
python reclassify_entities.py     # populates entity_roles for every article
cd ..
streamlit run dashboard.py
```

In the browser: open a Sports-topic article's detail page — the bottom-right
card should show Athletes / Teams / Major Events sections (Teams is new).
Open a non-Sports article — below the whole metrics grid, a new full-width
"Important Entities" section should appear with whichever categories matched
(Actors/Musicians/Politicians/Movies-TV/Organizations), each chip showing a
mention-count badge, and any chip whose entity got Wikipedia-enriched should
link out. If your dev data has an ambiguous name mentioned two ways (e.g.
"Sinner" and "Jannik Sinner" in the same article), confirm it appears as
**one** chip with the summed count, not two.

- [ ] **Step 5: Commit**

```bash
git add dash_pages/10_Article_Detail.py
git commit -m "Add Teams section to Sports Highlights and a full-width Important Entities widget for other articles"
```

---

## Self-Review Notes

- **Spec coverage:** A (taxonomy) → Task 1. B (cost cap) → Task 2. C (alias
  merge fix + schema) → Tasks 3 & 4. D (all articles) → Task 4. E (dashboard)
  → Task 5. All five spec sections have a task.
- **Placeholder scan:** none found — every step has complete, runnable code.
- **Type/name consistency:** `classify_entity_roles` (Task 2) is the name
  used consistently in Task 4's import and call site.
  `_merge_duplicate_urls` (Task 3) is used consistently inside
  `save_entity_roles` in the same task. `entity_roles` table's
  `mention_count` column (Task 4 migration) matches the column name used in
  `_merge_duplicate_urls`'s dict keys (Task 3) and `_role_chips`'s DataFrame
  column access (Task 5).
