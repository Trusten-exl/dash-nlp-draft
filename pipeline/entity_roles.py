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

CLAUDE_MODEL = "claude-opus-4-8"
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

# Positive label first; whichever description wins the zero-shot pass maps back
# to its key. A positive win only "sticks" if it clears CONFIDENCE.
PERSON_LABELS = {
    "athlete": "a professional athlete, sports player, or competitor",
    "other": "a person who is not an athlete",
}

EVENT_LABELS = {
    "sporting_event": "a sports competition, game, match, tournament, or championship",
    "other": "an event that is not a sporting event",
}

# Minimum probability for the positive label before we accept it. Tuned
# conservatively so the card errs toward precision (real athletes/games) over
# recall. Adjust after eyeballing reclassify output.
CONFIDENCE = 0.55


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
    descriptions = list(label_map.values())
    result = classifier(
        context,
        candidate_labels=descriptions,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    desc_to_key = {desc: key for key, desc in label_map.items()}
    return desc_to_key[result["labels"][0]], float(result["scores"][0])

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



def classify_sports_entities(article, entities, classifier=model):
    """
    Classify each PERSON / EVENT entity by its role in a sports article.

    Parameters
    ----------
    article : mapping with a "text" key (the article body for context).
    entities : iterable of mappings, each with "entity_text" and "entity_label".
    classifier : the shared zero-shot pipeline (defaults to the BART-MNLI model).

    Returns
    -------
    list of dicts: {entity_text, entity_label, role, confidence}
        role is "athlete", "sporting_event", or "other". Only PERSON entities
        can be "athlete" and only EVENT entities can be "sporting_event".
    """
    text = article.get("text") or ""
    out = []

    for ent in entities:
        etext = ent["entity_text"]
        elabel = ent["entity_label"]

        # context = info.get('summary')

        if elabel == "PERSON":
            context = _entity_context(text, etext)
            key, conf = _classify_entity(context, PERSON_LABELS, classifier)
            role = "athlete" if key == "athlete" and conf >= CONFIDENCE else "other"
        elif elabel == "EVENT":
            context = _entity_context(text, etext)
            key, conf = _classify_entity(context, EVENT_LABELS, classifier)
            role = "sporting_event" if key == "sporting_event" and conf >= CONFIDENCE else "other"
        else:
            role, conf = "other", 0.0

        
        info = get_celebrity_info(context, etext)
        url = info.get("url") if info else None


        out.append(
            {
                "entity_text": etext,
                "entity_label": elabel,
                "role": role,
                "url": url,
                "confidence": conf,
            }
        )

    return out
