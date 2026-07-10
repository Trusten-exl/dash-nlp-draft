from transformers import pipeline
from sic_data import DIVISIONS, SIC2, SIC3, SIC4

# pre-set intent names for use in classification - to be edited
# SIC_DIVISIONS = [

#     """
#     Agricultural production, farming, ranching, livestock, crop cultivation,
#     forestry, timber production, commercial fishing, aquaculture, agricultural
#     equipment and operations, or businesses primarily engaged in producing natural
#     food and biological resources. The article's central focus is on industries
#     that cultivate, harvest, or manage renewable natural resources rather than
#     manufacturing, processing, or retailing finished products.
#     """,

#     """
#     The exploration, extraction, drilling, quarrying, or production of natural
#     resources from the earth, including oil, natural gas, coal, metals,
#     minerals, stone, and other raw materials. The article primarily discusses
#     companies, operations, technology, regulations, or economic activity related
#     to resource extraction rather than manufacturing or distribution.
#     """,

#     """
#     The planning, development, renovation, or construction of residential,
#     commercial, industrial, or public infrastructure projects. The article focuses
#     on construction contractors, engineering firms, building materials, project
#     development, infrastructure investment, or businesses whose primary activity
#     is creating or improving physical structures.
#     """,

#     """
#     Businesses or industries that transform raw materials or components into
#     finished physical products through manufacturing, assembly, fabrication, or
#     industrial production. This includes automobiles, machinery, electronics,
#     consumer goods, pharmaceuticals, chemicals, food processing, aerospace,
#     industrial equipment, factories, and production facilities. The emphasis is on
#     making tangible products rather than providing services.
#     """,

#     """
#     The movement of people, goods, energy, information, or public resources.
#     Articles primarily discussing airlines, trucking, railroads, shipping,
#     logistics, ports, pipelines, public transit, electricity, water utilities,
#     natural gas distribution, telecommunications infrastructure, or utility
#     providers belong in this category.
#     """,

#     """
#     Businesses whose primary role is purchasing, storing, and distributing goods
#     to other businesses rather than directly to consumers. The article focuses on
#     wholesalers, commercial distributors, supply chains, industrial suppliers,
#     business-to-business commerce, inventory distribution, or wholesale markets.
#     """,

#     """
#     Businesses that sell products or services directly to individual consumers.
#     The article primarily discusses retailers, supermarkets, department stores,
#     restaurants, online retailers, consumer shopping behavior, retail operations,
#     store openings or closures, pricing, merchandising, or consumer-facing
#     commerce.
#     """,

#     """
#     Financial institutions, investment firms, insurance companies, banking,
#     capital markets, lending, mortgages, venture capital, asset management,
#     financial regulation, commercial or residential real estate, housing markets,
#     property development, or businesses whose primary purpose is managing money,
#     risk, investments, or property.
#     """,

#     """
#     Organizations whose primary business is providing professional, technical,
#     educational, healthcare, hospitality, entertainment, consulting, software,
#     information technology, legal, advertising, scientific, or other intangible
#     services rather than producing physical goods. The article focuses on service
#     delivery, expertise, customer support, digital products, healthcare,
#     education, tourism, or other service-based economic activities.
#     """,

#     """
#     Government agencies, regulatory organizations, public institutions, military
#     administration, public policy implementation, taxation, law enforcement,
#     government operations, or official governmental functions. The article's
#     primary focus is on the activities of government organizations acting in their
#     administrative or regulatory capacity rather than on private industry.
#     """
# ]


# Div_MAPPING = {

#     "Agriculture, Forestry, and Fishing":
#     """
#     Agricultural production, farming, ranching, livestock, crop cultivation,
#     forestry, timber production, commercial fishing, aquaculture, agricultural
#     equipment and operations, or businesses primarily engaged in producing natural
#     food and biological resources. The article's central focus is on industries
#     that cultivate, harvest, or manage renewable natural resources rather than
#     manufacturing, processing, or retailing finished products.
#     """,

#     "Mining":
#     """
#     The exploration, extraction, drilling, quarrying, or production of natural
#     resources from the earth, including oil, natural gas, coal, metals,
#     minerals, stone, and other raw materials. The article primarily discusses
#     companies, operations, technology, regulations, or economic activity related
#     to resource extraction rather than manufacturing or distribution.
#     """,

#     "Construction":
#     """
#     The planning, development, renovation, or construction of residential,
#     commercial, industrial, or public infrastructure projects. The article focuses
#     on construction contractors, engineering firms, building materials, project
#     development, infrastructure investment, or businesses whose primary activity
#     is creating or improving physical structures.
#     """,

#     "Manufacturing":
#     """
#     Businesses or industries that transform raw materials or components into
#     finished physical products through manufacturing, assembly, fabrication, or
#     industrial production. This includes automobiles, machinery, electronics,
#     consumer goods, pharmaceuticals, chemicals, food processing, aerospace,
#     industrial equipment, factories, and production facilities. The emphasis is on
#     making tangible products rather than providing services.
#     """,

#     "Transportation and Public Utilities":
#     """
#     The movement of people, goods, energy, information, or public resources.
#     Articles primarily discussing airlines, trucking, railroads, shipping,
#     logistics, ports, pipelines, public transit, electricity, water utilities,
#     natural gas distribution, telecommunications infrastructure, or utility
#     providers belong in this category.
#     """,

#     "Wholesale Trade":
#     """
#     Businesses whose primary role is purchasing, storing, and distributing goods
#     to other businesses rather than directly to consumers. The article focuses on
#     wholesalers, commercial distributors, supply chains, industrial suppliers,
#     business-to-business commerce, inventory distribution, or wholesale markets.
#     """,

#     "Retail Trade":
#     """
#     Businesses that sell products or services directly to individual consumers.
#     The article primarily discusses retailers, supermarkets, department stores,
#     restaurants, online retailers, consumer shopping behavior, retail operations,
#     store openings or closures, pricing, merchandising, or consumer-facing
#     commerce.
#     """,

#     "Finance, Insurance, and Real Estate":
#     """
#     Financial institutions, investment firms, insurance companies, banking,
#     capital markets, lending, mortgages, venture capital, asset management,
#     financial regulation, commercial or residential real estate, housing markets,
#     property development, or businesses whose primary purpose is managing money,
#     risk, investments, or property.
#     """,

#     "Services":
#     """
#     Organizations whose primary business is providing professional, technical,
#     educational, healthcare, hospitality, entertainment, consulting, software,
#     information technology, legal, advertising, scientific, or other intangible
#     services rather than producing physical goods. The article focuses on service
#     delivery, expertise, customer support, digital products, healthcare,
#     education, tourism, or other service-based economic activities.
#     """,

#     "Public Administration":
#     """
#     Government agencies, regulatory organizations, public institutions, military
#     administration, public policy implementation, taxation, law enforcement,
#     government operations, or official governmental functions. The article's
#     primary focus is on the activities of government organizations acting in their
#     administrative or regulatory capacity rather than on private industry.
#     """
# }

# # print(MAPPING)

# MAPPED = {v: k for k , v in Div_MAPPING.items()}

# selected model for selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

# def classify_article_division(article):
#     """
#     classifies industry mentioned in article
#     """
#     # create var classification tet ot assign what text will be used
#     classification_text=article['text'][:1000]

#     # get raw output in results
#     result = model(
#         classification_text,
#         candidate_labels=SIC_DIVISIONS,
#         hypothesis_template="The article focuses on {}",
#         multi_label=False
#     )

#     # sort into intents list for use in sql
#     division = []

#     for rank in range(len(result['labels'])):
#         division.append({
#             'division': MAPPED[result['labels'][rank]],
#             'confidence': float(result['scores'][rank]),
#             'rank': rank+1
#         })

#     return division



# ------------------------------------------------------------
# Industry-relevance gate
# ------------------------------------------------------------

# Contrasting hypotheses used to decide whether an article is about an
# industry at all, before we try to place it in the SIC hierarchy.
_RELEVANCE_LABELS = {
    "industry": (
        "a specific industry, company, product, or commercial business activity"
    ),
    "general": (
        "general news such as politics, government, sports, crime, or human "
        "interest, not centered on any particular industry"
    ),
}

HYPOTHESIS_TEMPLATE = "This article is primarily about {}."

# Tunable gates.
RELEVANCE_THRESHOLD = 0.55   # min P(industry-related) to attempt SIC at all
DIVISION_THRESHOLD = 0.25    # min softmax score for the winning division

# Concise, mutually-distinct division hypotheses. The SIC division names in
# sic_data embed ~450-char descriptions, which overlap heavily and make the
# zero-shot model default to whichever is listed first (Agriculture). Short
# discriminative phrases classify far better; codes map back to the real names.
DIVISION_HYPOTHESES = {
    "A": "agriculture, farming, ranching, forestry, or fishing",
    "B": "mining, quarrying, or oil and gas extraction",
    "C": "construction of buildings or infrastructure",
    "D": "manufacturing physical products such as machinery, electronics, vehicles, chemicals, or food",
    "E": "transportation, utilities, energy, or telecommunications",
    "F": "wholesale distribution of goods to businesses",
    "G": "retail stores selling goods to consumers",
    "H": "finance, banking, insurance, investing, or real estate",
    "I": "services such as software, technology, healthcare, education, media, legal, or hospitality",
    "J": "government, public administration, law enforcement, or the military",
}


def build_sic_input(article, max_chars=1500):
    """Combine the most informative fields for classification."""
    parts = [
        article.get("title") or "",
        article.get("description") or "",
        article.get("text") or "",
    ]
    combined = " ".join(p.strip() for p in parts if p and p.strip())
    return combined[:max_chars]


def is_industry_related(text, classifier, threshold=0.55):
    """
    Decide whether an article is about an industry at all.

    Runs a two-way zero-shot contrast (industry vs. general news) and returns
    the probability that the article is industry-focused plus a boolean gate.
    """
    result = classifier(
        text,
        candidate_labels=list(_RELEVANCE_LABELS.values()),
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    scores = dict(zip(result["labels"], result["scores"]))
    industry_prob = float(scores[_RELEVANCE_LABELS["industry"]])

    return {"related": industry_prob >= threshold, "score": industry_prob}


# ------------------------------------------------------------
# Hierarchical SIC classification
# ------------------------------------------------------------
def predict_level(text, candidates, classifier, top_k=3,
                  multi_label=False, hypothesis_template=None):
    """
    Generic zero-shot prediction for a hierarchy level.

    Parameters
    ----------
    text : str
        Article text.

    candidates : dict
        Dictionary of candidate codes.

    classifier
        HuggingFace zero-shot pipeline.

    top_k : int
        Number of predictions to return.

    multi_label : bool
        When True each label is scored independently (absolute entailment
        probability); when False the labels compete via softmax.

    hypothesis_template : str | None
        Optional NLI hypothesis template.

    Returns
    -------
    list
    """

    labels = [info["name"] for info in candidates.values()]

    kwargs = {"multi_label": multi_label}
    if hypothesis_template:
        kwargs["hypothesis_template"] = hypothesis_template

    results = classifier(
        text,
        candidate_labels=labels,
        **kwargs,
    )

    predictions = []

    for label, score in zip(results["labels"], results["scores"]):

        for code, info in candidates.items():

            if info["name"] == label:

                predictions.append(
                    {
                        "code": code,
                        "name": info["name"].split(':')[0],
                        "score": float(score),
                    }
                )

                break

    return predictions[:top_k]


def get_sic2_candidates(division_code):
    """
    Returns all 2-digit SIC codes under a division.
    """

    return {
        code: SIC2[code]
        for code in DIVISIONS[division_code]["children"]
    }


def get_sic3_candidates(sic2_code):
    """
    Returns all 3-digit SIC codes under a 2-digit SIC.
    """

    return {
        code: SIC3[code]
        for code in SIC2[sic2_code]["children"]
    }


def get_sic4_candidates(sic3_code):
    """
    Returns all 4-digit SIC codes under a 3-digit SIC.
    """

    return {
        code: SIC4[code]
        for code in SIC3[sic3_code]["children"]
    }


def classify_division(text, classifier, top_k=3):
    """
    Classify the SIC division using concise discriminative hypotheses.

    Uses softmax (multi_label=False) to pick the single best-fitting division.
    Non-industry articles are already filtered by the relevance gate upstream,
    so here we just need a clear winner (the concise labels prevent the old
    default-to-Agriculture behaviour); DIVISION_THRESHOLD is a light floor.
    """
    labels = list(DIVISION_HYPOTHESES.values())
    label_to_code = {v: k for k, v in DIVISION_HYPOTHESES.items()}

    results = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    predictions = []
    for label, score in zip(results["labels"], results["scores"]):
        code = label_to_code[label]
        predictions.append(
            {
                "code": code,
                "name": DIVISIONS[code]["name"].split(":")[0],
                "score": float(score),
            }
        )

    return predictions[:top_k]


def classify_sic2(text, division_code, classifier, top_k=3):

    candidates = get_sic2_candidates(division_code)

    return predict_level(
        text=text,
        candidates=candidates,
        classifier=classifier,
        top_k=top_k,
    )


def classify_sic3(text, sic2_code, classifier, top_k=3):

    candidates = get_sic3_candidates(sic2_code)

    return predict_level(
        text=text,
        candidates=candidates,
        classifier=classifier,
        top_k=top_k,
    )


def classify_sic4(text, sic3_code, classifier, top_k=3):

    candidates = get_sic4_candidates(sic3_code)

    return predict_level(
        text=text,
        candidates=candidates,
        classifier=classifier,
        top_k=top_k,
    )


def classify_sic_article(article, classifier=model, top_k=3,
                         relevance_threshold=RELEVANCE_THRESHOLD,
                         division_threshold=DIVISION_THRESHOLD):
    """
    Full hierarchical SIC classification, gated by industry relevance.

    Returns
    -------
    dict
        {
          "related": bool,                # is the article about an industry?
          "relevance_score": float,       # P(industry-related)
          "predictions": {                # empty lists when not related
              "division": [...], "sic2": [...],
              "sic3": [...], "sic4": [...],
          },
        }
    """

    text = build_sic_input(article)

    empty = {"division": [], "sic2": [], "sic3": [], "sic4": []}

    def result(related, reason, relevance, top_division=None, predictions=empty):
        return {
            "related": related,
            "reason": reason,
            "relevance_score": relevance["score"],
            "top_division": top_division,   # {name, score} if a division was scored
            "predictions": predictions,
        }

    # 1) Is this even about an industry?
    relevance = is_industry_related(text, classifier, relevance_threshold)
    if not relevance["related"]:
        return result(False, "not_industry", relevance)

    # 2) Which division — with a confidence guard and an explicit block on the
    #    catch-all "Nonclassifiable Establishments" division.
    division_predictions = classify_division(text, classifier, top_k=top_k)
    best = division_predictions[0] if division_predictions else None
    top_division = {"name": best["name"], "score": best["score"]} if best else None

    if best is None:
        return result(False, "no_division", relevance, top_division)
    if best["code"] == "K":
        return result(False, "nonclassifiable", relevance, top_division)
    if best["score"] < division_threshold:
        return result(False, "weak_division", relevance, top_division)

    best_division = best["code"]

    sic2_predictions = classify_sic2(text, best_division, classifier, top_k=top_k)
    best_sic2 = sic2_predictions[0]["code"]

    sic3_predictions = classify_sic3(text, best_sic2, classifier, top_k=top_k)
    best_sic3 = sic3_predictions[0]["code"]

    sic4_predictions = classify_sic4(text, best_sic3, classifier, top_k=top_k)

    return result(
        True,
        "ok",
        relevance,
        top_division,
        {
            "division": division_predictions,
            "sic2": sic2_predictions,
            "sic3": sic3_predictions,
            "sic4": sic4_predictions,
        },
    )

