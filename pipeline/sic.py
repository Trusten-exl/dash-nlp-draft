from transformers import pipeline
from sic_data import DIVISIONS, SIC2, SIC3, SIC4

# pre-set intent names for use in classification - to be edited
SIC_DIVISIONS = [

    """
    Agricultural production, farming, ranching, livestock, crop cultivation,
    forestry, timber production, commercial fishing, aquaculture, agricultural
    equipment and operations, or businesses primarily engaged in producing natural
    food and biological resources. The article's central focus is on industries
    that cultivate, harvest, or manage renewable natural resources rather than
    manufacturing, processing, or retailing finished products.
    """,

    """
    The exploration, extraction, drilling, quarrying, or production of natural
    resources from the earth, including oil, natural gas, coal, metals,
    minerals, stone, and other raw materials. The article primarily discusses
    companies, operations, technology, regulations, or economic activity related
    to resource extraction rather than manufacturing or distribution.
    """,

    """
    The planning, development, renovation, or construction of residential,
    commercial, industrial, or public infrastructure projects. The article focuses
    on construction contractors, engineering firms, building materials, project
    development, infrastructure investment, or businesses whose primary activity
    is creating or improving physical structures.
    """,

    """
    Businesses or industries that transform raw materials or components into
    finished physical products through manufacturing, assembly, fabrication, or
    industrial production. This includes automobiles, machinery, electronics,
    consumer goods, pharmaceuticals, chemicals, food processing, aerospace,
    industrial equipment, factories, and production facilities. The emphasis is on
    making tangible products rather than providing services.
    """,

    """
    The movement of people, goods, energy, information, or public resources.
    Articles primarily discussing airlines, trucking, railroads, shipping,
    logistics, ports, pipelines, public transit, electricity, water utilities,
    natural gas distribution, telecommunications infrastructure, or utility
    providers belong in this category.
    """,

    """
    Businesses whose primary role is purchasing, storing, and distributing goods
    to other businesses rather than directly to consumers. The article focuses on
    wholesalers, commercial distributors, supply chains, industrial suppliers,
    business-to-business commerce, inventory distribution, or wholesale markets.
    """,

    """
    Businesses that sell products or services directly to individual consumers.
    The article primarily discusses retailers, supermarkets, department stores,
    restaurants, online retailers, consumer shopping behavior, retail operations,
    store openings or closures, pricing, merchandising, or consumer-facing
    commerce.
    """,

    """
    Financial institutions, investment firms, insurance companies, banking,
    capital markets, lending, mortgages, venture capital, asset management,
    financial regulation, commercial or residential real estate, housing markets,
    property development, or businesses whose primary purpose is managing money,
    risk, investments, or property.
    """,

    """
    Organizations whose primary business is providing professional, technical,
    educational, healthcare, hospitality, entertainment, consulting, software,
    information technology, legal, advertising, scientific, or other intangible
    services rather than producing physical goods. The article focuses on service
    delivery, expertise, customer support, digital products, healthcare,
    education, tourism, or other service-based economic activities.
    """,

    """
    Government agencies, regulatory organizations, public institutions, military
    administration, public policy implementation, taxation, law enforcement,
    government operations, or official governmental functions. The article's
    primary focus is on the activities of government organizations acting in their
    administrative or regulatory capacity rather than on private industry.
    """
]

Div_MAPPING = {

    "Agriculture, Forestry, and Fishing":
    """
    Agricultural production, farming, ranching, livestock, crop cultivation,
    forestry, timber production, commercial fishing, aquaculture, agricultural
    equipment and operations, or businesses primarily engaged in producing natural
    food and biological resources. The article's central focus is on industries
    that cultivate, harvest, or manage renewable natural resources rather than
    manufacturing, processing, or retailing finished products.
    """,

    "Mining":
    """
    The exploration, extraction, drilling, quarrying, or production of natural
    resources from the earth, including oil, natural gas, coal, metals,
    minerals, stone, and other raw materials. The article primarily discusses
    companies, operations, technology, regulations, or economic activity related
    to resource extraction rather than manufacturing or distribution.
    """,

    "Construction":
    """
    The planning, development, renovation, or construction of residential,
    commercial, industrial, or public infrastructure projects. The article focuses
    on construction contractors, engineering firms, building materials, project
    development, infrastructure investment, or businesses whose primary activity
    is creating or improving physical structures.
    """,

    "Manufacturing":
    """
    Businesses or industries that transform raw materials or components into
    finished physical products through manufacturing, assembly, fabrication, or
    industrial production. This includes automobiles, machinery, electronics,
    consumer goods, pharmaceuticals, chemicals, food processing, aerospace,
    industrial equipment, factories, and production facilities. The emphasis is on
    making tangible products rather than providing services.
    """,

    "Transportation and Public Utilities":
    """
    The movement of people, goods, energy, information, or public resources.
    Articles primarily discussing airlines, trucking, railroads, shipping,
    logistics, ports, pipelines, public transit, electricity, water utilities,
    natural gas distribution, telecommunications infrastructure, or utility
    providers belong in this category.
    """,

    "Wholesale Trade":
    """
    Businesses whose primary role is purchasing, storing, and distributing goods
    to other businesses rather than directly to consumers. The article focuses on
    wholesalers, commercial distributors, supply chains, industrial suppliers,
    business-to-business commerce, inventory distribution, or wholesale markets.
    """,

    "Retail Trade":
    """
    Businesses that sell products or services directly to individual consumers.
    The article primarily discusses retailers, supermarkets, department stores,
    restaurants, online retailers, consumer shopping behavior, retail operations,
    store openings or closures, pricing, merchandising, or consumer-facing
    commerce.
    """,

    "Finance, Insurance, and Real Estate":
    """
    Financial institutions, investment firms, insurance companies, banking,
    capital markets, lending, mortgages, venture capital, asset management,
    financial regulation, commercial or residential real estate, housing markets,
    property development, or businesses whose primary purpose is managing money,
    risk, investments, or property.
    """,

    "Services":
    """
    Organizations whose primary business is providing professional, technical,
    educational, healthcare, hospitality, entertainment, consulting, software,
    information technology, legal, advertising, scientific, or other intangible
    services rather than producing physical goods. The article focuses on service
    delivery, expertise, customer support, digital products, healthcare,
    education, tourism, or other service-based economic activities.
    """,

    "Public Administration":
    """
    Government agencies, regulatory organizations, public institutions, military
    administration, public policy implementation, taxation, law enforcement,
    government operations, or official governmental functions. The article's
    primary focus is on the activities of government organizations acting in their
    administrative or regulatory capacity rather than on private industry.
    """
}

# print(MAPPING)

MAPPED = {v: k for k , v in Div_MAPPING.items()}

# selected model for selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_article_division(article):
    """
    classifies industry mentioned in article
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=SIC_DIVISIONS,
        hypothesis_template="The article focuses on {}",
        multi_label=False
    )

    # sort into intents list for use in sql
    division = []

    for rank in range(len(result['labels'])):
        division.append({
            'division': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return division



# rewriting with hierarchal structure
def predict_level(text, candidates, classifier, top_k=3):
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

    Returns
    -------
    list
    """

    labels = [info["name"] for info in candidates.values()]

    results = classifier(
        text,
        candidate_labels=labels,
        multi_label=False,
    )

    predictions = []

    for label, score in zip(results["labels"], results["scores"]):

        for code, info in candidates.items():

            if info["name"] == label:

                predictions.append(
                    {
                        "code": code,
                        "name": info["name"],
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

    return predict_level(
        text=text,
        candidates=DIVISIONS,
        classifier=classifier,
        top_k=top_k,
    )


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


def classify_sic_article(article, classifier=model, top_k=3):
    """
    Full hierarchical SIC classification.
    """

    text = article['text'][:1000]

    division_predictions = classify_division(
        text,
        classifier,
        top_k=top_k,
    )

    best_division = division_predictions[0]["code"]

    sic2_predictions = classify_sic2(
        text,
        best_division,
        classifier,
        top_k=top_k,
    )

    best_sic2 = sic2_predictions[0]["code"]

    sic3_predictions = classify_sic3(
        text,
        best_sic2,
        classifier,
        top_k=top_k,
    )

    best_sic3 = sic3_predictions[0]["code"]

    sic4_predictions = classify_sic4(
        text,
        best_sic3,
        classifier,
        top_k=top_k,
    )

    return {
        "division": division_predictions,
        "sic2": sic2_predictions,
        "sic3": sic3_predictions,
        "sic4": sic4_predictions,
    }

