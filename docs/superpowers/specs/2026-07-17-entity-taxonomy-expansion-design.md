# Entity taxonomy expansion + all-article classification

Date: 2026-07-17
Status: approved (design sections A–E confirmed by user; user waived spec/plan review — proceed straight through to implementation)

## Context

`pipeline/entity_roles.py` currently classifies only PERSON/EVENT entities in
Sports-topic articles into a binary role per label (`athlete`/`other`,
`sporting_event`/`other`), then enriches matched entities with a Wikipedia
link via Claude-powered query generation + candidate resolution
(`get_celebrity_info`). The dashboard's "Sports Highlights" card
(`dash_pages/10_Article_Detail.py`) is the only place this data is shown, and
only for Sports-topic articles.

The user wants this generalized: recognize more categories (actors, singers,
sports teams, movies/TV shows, politicians) across **all** articles, not just
Sports, and surface the result in a new dashboard widget for non-sports
articles (Sports keeps its existing card, extended with a Teams section).

Two things make this non-trivial:

1. **Cost.** Enrichment is 2 Claude calls + 2 Wikipedia calls per entity. Going
   from "PERSON/EVENT in Sports articles" to "7 categories across every
   article" multiplies volume substantially — directly opposed to the cost
   cut just made (skipping enrichment for `role == "other"`).
2. **Alias merging.** NER dedups by exact text only. "Sinner" and "Jannik
   Sinner" are two separate rows today. Investigating this surfaced an
   existing bug: `save.py::save_entity_roles` already groups duplicate
   entities by resolved Wikipedia `url` and keeps the longer name, but it
   **drops** the shorter alias's row instead of merging its mention count —
   so today those mentions are silently lost, not double-counted.

## Design

### A. Taxonomy & classification

Extend `_classify_entity` usage (the function itself is unchanged — already
generic over any `label -> description` map) from today's binary
"one positive label vs. other" to real multi-way category sets per NER label.
One zero-shot call per entity picks the best of N candidates — same cost as
today's binary call, just a richer label set:

```python
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
EVENT_LABELS = {  # unchanged
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
```

`pipeline/ner.py`'s `KEEP_LABELS` gains `"WORK_OF_ART"` — movies/TV shows
aren't extracted by NER at all today. (Caveat: this only affects articles
processed *after* the change; existing articles' `entities` rows won't
retroactively gain WORK_OF_ART entities unless NER is re-run for them. Out of
scope for this spec — noted for awareness.)

### B. Cost control: classify cheap, enrich capped

Role classification (local BART-MNLI, already loaded) is cheap — it runs on
every matched entity (any label in `LABEL_MAPS`) in every article now, not
just Sports/PERSON/EVENT. The expensive part (`get_celebrity_info`'s 2 Claude
+ 2 Wikipedia calls) only runs for entities that:

1. classified to a real category (`role != "other"`), **and**
2. rank in the **top `ENRICH_CAP` (20) by raw `mention_count`** among such
   entities, per article.

Entities outside the cap keep their role label (so they still count toward
mention totals and show up in raw form) but get no Wikipedia link
(`url = None`), rendered as a plain (non-linked) chip.

### C. Alias merging, properly

`save_entity_roles` already groups by resolved `url` and keeps the longer
name as canonical (`save.py:389-394`). The fix: **sum `mention_count`** across
merged aliases instead of discarding the dropped one. Requires:

- `mention_count` threaded from the `entities` table through
  `classify_sports_entities`'s per-entity dict into `save_entity_roles`.
- New column `entity_roles.mention_count INTEGER`, added via
  `ALTER TABLE ... ADD COLUMN` guarded by a `PRAGMA table_info` check (the
  existing `CREATE TABLE IF NOT EXISTS` in `reclassify_entities.py::ensure_tables`
  won't add a column to an already-existing table on the user's dev DB).

### D. Run it on all articles

`reclassify_entities.py`'s `sports_article_ids()` is replaced by an
"all articles" query (`SELECT article_id FROM articles ORDER BY article_id`).
The entities query gains `ORG`/`WORK_OF_ART` to its label filter and adds
`mention_count` to the `SELECT`. Renamed to reflect the broadened scope (e.g.
`classifiable_article_ids`).

The dashboard already determines Sports-vs-not independently at render time
via `is_sports_article(topics)` — the pipeline doesn't need to know or care;
it just classifies+stores roles for every article, and the dashboard decides
how to display them.

### E. Dashboard

Simplification enabled by C: since `entity_roles` rows are now the
already-merged, one-row-per-real-entity source of truth (url + mention_count
together), the chip renderer can read directly from `entity_roles` instead of
joining against raw `entities` + a separate allow-list. Replaces
`_entity_chips`/`_allowed_entities` for role-based rendering with a single
`_role_chips(roles_df, role_name, accent, limit)` that filters `roles_df` by
`role`, sorts by `mention_count` desc, and renders each row as a linked chip
(if `url` is set) or plain chip (if not, i.e. outside the enrichment cap) with
a count badge (re-enabling the currently-hardcoded-off badge, now driven by
the merged count).

- **Sports Highlights card** (bottom-right, sports articles): same shape,
  gains a **Teams** section (`sports_team` role, ORG label), alongside the
  existing Athletes/Major Events sections.
- **New full-width "Important Entities" section**, rendered below the entire
  2×2 grid, for non-sports articles only: grouped sub-sections per category
  (Actors, Musicians, Politicians, Movies/TV Shows, Organizations), each a row
  of chips via `_role_chips`.

## Out of scope

- Backfilling `WORK_OF_ART` entities for already-processed articles (NER
  would need to be re-run; not requested).
- Any category beyond the 7 confirmed (athlete, actor, musician, politician,
  sports_team, sporting_event, movie_or_tv_show) — easy to add later since
  each is one more entry in a label map.
- Changing the enrichment cap's value beyond a sensible default (20) — easy to
  tune later as a single constant.

## Verification

- `pipeline/test_entity_roles.py` gets a new assert case (or a sibling test)
  for the merge-summing logic in `save_entity_roles`.
- Manual run: `python reclassify_entities.py` against the dev DB, then
  `streamlit run dashboard.py` — check a Sports article's card shows a Teams
  section, and a non-sports article (e.g. entertainment/politics topic) shows
  the new full-width widget with sensible categories and merged counts (e.g.
  a "Sinner"/"Jannik Sinner" case, if present in the dev data, should show as
  one chip with a summed count).
