from transformers import pipeline


# pre-set intent names for use in classification - to be edited
ARTICLE_INTENTS = [
    "A factual news article whose primary purpose is to inform readers about recent events, announcements, or developments. The article focuses on presenting verified information, describing what happened, who was involved, when and where events occurred, and any relevant context. It aims to remain objective and balanced, relying on evidence, official statements, eyewitness accounts, or documented facts rather than personal interpretation. The emphasis is on reporting events accurately rather than explaining deeper causes or expressing opinions.",

    "An article primarily intended to persuade, argue, or express a viewpoint. The author presents personal judgments, interpretations, recommendations, or critiques rather than simply reporting facts. The writing may use evidence or examples to support its position, but the central goal is to convince readers or advocate for a particular perspective, policy, belief, or course of action. Subjective language, editorial commentary, and personal reasoning are common characteristics",

    "An article whose primary goal is to help readers understand a topic, concept, event, process, or issue. Rather than focusing on breaking news or personal opinions, it provides background information, definitions, historical context, step-by-step explanations, or answers to common questions. The emphasis is on education and clarity, making complex subjects easier to understand for a general audience.",

    "An article that examines the significance, causes, implications, or broader meaning of events or issues. It goes beyond simply reporting facts by interpreting evidence, identifying patterns or trends, comparing different viewpoints, evaluating consequences, or discussing possible future outcomes. While generally grounded in evidence, the focus is on thoughtful interpretation and insight rather than advocacy or straightforward explanation.",

    "An instructional article designed to help readers accomplish a task or solve a problem. It provides practical advice, recommendations, step-by-step instructions, best practices, or actionable tips. The primary purpose is to enable readers to perform an activity, make a decision, or improve a skill, rather than to report news or analyze events.",

    "An article that evaluates or critiques a product, service, book, movie, restaurant, technology, event, or other item based on experience or established criteria. It discusses strengths, weaknesses, quality, performance, or value, often leading to an overall recommendation or judgment. While factual information may be included, the central purpose is assessment and evaluation rather than reporting or explaining.",
]

MAPPING = {
    "Report": "A factual news article whose primary purpose is to inform readers about recent events, announcements, or developments. The article focuses on presenting verified information, describing what happened, who was involved, when and where events occurred, and any relevant context. It aims to remain objective and balanced, relying on evidence, official statements, eyewitness accounts, or documented facts rather than personal interpretation. The emphasis is on reporting events accurately rather than explaining deeper causes or expressing opinions.",

    "Opinion": "An article primarily intended to persuade, argue, or express a viewpoint. The author presents personal judgments, interpretations, recommendations, or critiques rather than simply reporting facts. The writing may use evidence or examples to support its position, but the central goal is to convince readers or advocate for a particular perspective, policy, belief, or course of action. Subjective language, editorial commentary, and personal reasoning are common characteristics",

    "Explainer": "An article whose primary goal is to help readers understand a topic, concept, event, process, or issue. Rather than focusing on breaking news or personal opinions, it provides background information, definitions, historical context, step-by-step explanations, or answers to common questions. The emphasis is on education and clarity, making complex subjects easier to understand for a general audience.",

    "Analysis": "An article that examines the significance, causes, implications, or broader meaning of events or issues. It goes beyond simply reporting facts by interpreting evidence, identifying patterns or trends, comparing different viewpoints, evaluating consequences, or discussing possible future outcomes. While generally grounded in evidence, the focus is on thoughtful interpretation and insight rather than advocacy or straightforward explanation.",

    "Guide": "An instructional article designed to help readers accomplish a task or solve a problem. It provides practical advice, recommendations, step-by-step instructions, best practices, or actionable tips. The primary purpose is to enable readers to perform an activity, make a decision, or improve a skill, rather than to report news or analyze events.",

    "Review": "An article that evaluates or critiques a product, service, book, movie, restaurant, technology, event, or other item based on experience or established criteria. It discusses strengths, weaknesses, quality, performance, or value, often leading to an overall recommendation or judgment. While factual information may be included, the central purpose is assessment and evaluation rather than reporting or explaining.",
}

# print(MAPPING)

MAPPED = {v: k for k , v in MAPPING.items()}

# selected model for intent selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_article_intent(article):
    """
    classifies intent of article using above selected model / intent
    returns dict of intent / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=ARTICLE_INTENTS,
        hypothesis_template="The intent of this news article is {}",
        multi_label=False
    )

    # sort into intents list for use in sql
    intents = []

    for rank in range(len(result['labels'])):
        intents.append({
            'intent': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return intents

