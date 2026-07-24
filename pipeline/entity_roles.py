# ============================================================
# entity_roles.py — classify sports entities via zero-shot NLP
# ============================================================
# The article-detail "Sports Highlights" card should list real athletes and
# real games/events, not every PERSON / EVENT that spaCy tagged (NER also
# catches coaches, brands, mislabelled team names, generic dates, etc.).
#
# This module uses zero-shot MNLI to decide, per entity:
#   * PERSON -> is this an athlete/competitor, or not?
#   * EVENT  -> is this a sporting event/game, or not?
#
# It reuses the shared BART-MNLI pipeline already loaded in sic.py so we don't
# hold a second ~1.6GB model in memory. Roles are precomputed here and stored
# in the DB (see save.save_entity_roles); the dashboard only reads them.

import re

from sic import model
import requests
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

CLAUDE_MODEL = "claude-haiku-4-5"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (via .env or the environment)


class WikiQuery(BaseModel):
    query: str


class EntityResolution(BaseModel):
    title: str | None  # None = no candidate is a good match


def resolve_entity(entity, context, results):
    """Ask Claude to pick the correct candidate title; verified against `results` by the caller."""
    candidates = "\n\n".join(
        f"Candidate {i+1}:\nTitle: {r['title']}\nSnippet: {r.get('snippet', '')[:300]}"
        for i, r in enumerate(results)
    )

    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Choose the single best Wikipedia article for this entity from the "
                "candidates below, using the context to disambiguate. Choose ONLY from "
                "the candidate titles verbatim, or null if none is a good match.\n\n"
                f"Entity: {entity}\nContext: {context}\n\nCandidates:\n{candidates}"
            ),
        }],
        output_format=EntityResolution,
    )
    return response.parsed_output.title


def generate_wiki_query(entity, context):
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Generate a short Wikipedia search query that will identify the correct "
                "entity. Always include the original entity name; add only the most "
                "useful identifying terms (profession, organization, sport, location, "
                f"event).\n\nEntity: {entity}\nContext: {context}"
            ),
        }],
        output_format=WikiQuery,
    )
    query = response.parsed_output.query.strip()
    return query or entity


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
    "executive": "a business executive, founder, or corporate leader",
    "other": "a person who is none of the above",
}

ORG_LABELS = {
    "sports_team": "a professional or amateur sports team or club",
    "company": "a business or company",
    "other": "an organization that is not a sports team or company",
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

# Minimum probability for the winning label before we accept it. Lowered from
# the original 0.55: gt_comparison.csv showed 143/675 GT entities (21%) landed
# on "other" while a real role was in fact the top candidate, just under the
# old bar - since "other" only ever comes from `role` itself winning the vote
# (see _classify_entity/classify_entity_roles below), lowering this can only
# recover suppressed real roles, never turn a genuine "other" into a false
# positive. Re-check gt_comparison.csv's entity recall/precision after
# lowering this to confirm the trade is worth it.
CONFIDENCE = 0.45


def _entity_context(text: str, entity_text: str, max_chars: int = 500) -> str:
    """
    Bare names ("Ronaldo") give MNLI little to work with, so we feed it the
    sentences from the article that actually mention the entity. Falls back to
    the raw name when no sentence matches (or there's no article text).
    """
    if not text:
        return entity_text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    needle = entity_text.lower()
    hits = [s.strip() for s in sentences if needle in s.lower()]
    context = " ".join(hits)[:max_chars]
    return context or entity_text


def _classify_entity(context: str, label_map: dict, classifier) -> tuple[str, float]:
    """Zero-shot over a label map's descriptions; returns (winning key, prob)."""
    return _classify_entities_batch([context], label_map, classifier)[0]


def _classify_entities_batch(contexts: list, label_map: dict, classifier) -> list:
    """
    Batched version of _classify_entity: every context shares the same
    label_map (and thus the same candidate set), so they classify in one
    call instead of one call per entity - an article with dozens of entities
    of the same type was previously one model call each.

    Returns a list of (winning key, prob) tuples, same order as `contexts`.
    """
    if not contexts:
        return []
    descriptions = list(label_map.values())
    desc_to_key = {desc: key for key, desc in label_map.items()}
    results = classifier(
        contexts,
        candidate_labels=descriptions,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    return [(desc_to_key[r["labels"][0]], float(r["scores"][0])) for r in results]

def _verified_title(title, results):
    """None means "no good match" (honored); a title not in `results` is a
    hallucination, so fall back to the naive top search hit instead of trusting it."""
    if title is None:
        return None
    if title in {r["title"] for r in results}:
        return title
    return results[0]["title"]


HEADERS = {"User-Agent": "news-dashboard/1.0 (contact: you@yourcompany.com)"}

def get_celebrity_info(context, name: str) -> dict:
    #Use LLM to determine best Query
    print(f'Starting {name}, generating query')
    query = generate_wiki_query(entity=name, context=context )
    print(f"Query: {query}, starting search")
    # Step 1: search for the closest matching title
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            },
            headers=HEADERS,
            timeout=5,
        )
        search_resp.raise_for_status()

    except (requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.RequestException) as e:
        print(f"Wikipedia request failed for '{name}': {e}")
        return {}
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return {"name": name, "error": "not found"}

    print("Resolving Title")
    title = _verified_title(resolve_entity(entity=name, context=context, results=results), results)
    if title is None:
        return {"name": name, "error": "no match"}
    print(f'Title: {title}, extracting data')

    # Step 2: get the summary + url for that title
    try:
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers=HEADERS,
            timeout=5,
        )
        summary_resp.raise_for_status()
    except (requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.RequestException) as e:
        print(f"Wikipedia request failed for '{name}': {e}")
        return{}
    
    data = summary_resp.json()
    # print(data)
    print(f'Finished {name}')
    return {
        "name": data.get("title"),
        "summary": data.get("extract"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }



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
    entities = list(entities)

    # Group by entity_label first: entities sharing a label_map share the
    # same candidate set, so each group classifies in one batched call
    # instead of one call per entity (an article can have dozens of PERSON
    # entities alone).
    by_label = {}
    for i, ent in enumerate(entities):
        by_label.setdefault(ent["entity_label"], []).append(i)

    contexts = [None] * len(entities)
    key_conf = [None] * len(entities)

    for elabel, idxs in by_label.items():
        label_map = LABEL_MAPS.get(elabel)
        if label_map is None:
            for i in idxs:
                contexts[i] = entities[i]["entity_text"]
                key_conf[i] = ("other", 0.0)
            continue

        for i in idxs:
            contexts[i] = _entity_context(text, entities[i]["entity_text"])
        batch = _classify_entities_batch([contexts[i] for i in idxs], label_map, classifier)
        for i, kc in zip(idxs, batch):
            key_conf[i] = kc

    classified = []
    for ent, context, (key, conf) in zip(entities, contexts, key_conf):
        role = key if conf >= CONFIDENCE else "other"
        classified.append({
            "entity_text": ent["entity_text"],
            "entity_label": ent["entity_label"],
            "role": role,
            "confidence": conf,
            "mention_count": ent.get("mention_count") or 1,
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
