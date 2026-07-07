from transformers import pipeline


# pre-set intent names for use in classification - to be edited
FORMATS = [
    "A time-sensitive news article published shortly after an event occurs. Its primary purpose is to quickly report new information, recent developments, or urgent updates. The article focuses on delivering the latest verified facts and may be updated as additional details become available. It emphasizes immediacy and timeliness rather than in-depth background, analysis, or storytelling.",

    "An in-depth article that explores a person, event, trend, or topic through detailed reporting, storytelling, interviews, or human-interest elements. The article emphasizes narrative, context, and descriptive writing rather than immediate news. It often examines broader themes, personal experiences, or unique perspectives to provide readers with a richer understanding of the subject.",

    "An article that interprets or evaluates events rather than simply reporting them. It examines causes, significance, strategies, trends, implications, or possible future outcomes. The article synthesizes facts and evidence to provide insight and context, helping readers understand why an event matters rather than only describing what happened.",

    "An article summarizing the events and outcome of a sporting contest. It describes the final score, key plays, standout performances, important statistics, turning points, and notable moments from the game. The primary purpose is to document and explain how the competition unfolded rather than discussing broader issues surrounding the teams or sport.",

    "A sports-related news article that focuses on events occurring outside of actual competition. Topics may include player transfers, injuries, coaching changes, contracts, league policies, business operations, legal matters, disciplinary actions, ownership, facilities, or other organizational developments. The article reports news connected to sports without primarily describing the action or results of a game.",

    "A standard news article whose primary purpose is to report factual information about an event, development, announcement, or situation in a clear, objective, and balanced manner. The article presents verified facts, statements from relevant sources, and essential context without emphasizing urgency, in-depth storytelling, personal opinion, or extensive interpretation. Unlike breaking news, it is not defined by immediacy; unlike a feature, it does not focus on narrative or human-interest storytelling; and unlike analysis, it does not primarily interpret the significance or implications of events. Its main goal is to accurately inform readers about what happened."
]

MAPPING = {
    "Breaking": "A time-sensitive news article published shortly after an event occurs. Its primary purpose is to quickly report new information, recent developments, or urgent updates. The article focuses on delivering the latest verified facts and may be updated as additional details become available. It emphasizes immediacy and timeliness rather than in-depth background, analysis, or storytelling.",

    "Feature": "An in-depth article that explores a person, event, trend, or topic through detailed reporting, storytelling, interviews, or human-interest elements. The article emphasizes narrative, context, and descriptive writing rather than immediate news. It often examines broader themes, personal experiences, or unique perspectives to provide readers with a richer understanding of the subject.",

    "Analysis": "An article that interprets or evaluates events rather than simply reporting them. It examines causes, significance, strategies, trends, implications, or possible future outcomes. The article synthesizes facts and evidence to provide insight and context, helping readers understand why an event matters rather than only describing what happened.",

    "Game Recap": "An article summarizing the events and outcome of a sporting contest. It describes the final score, key plays, standout performances, important statistics, turning points, and notable moments from the game. The primary purpose is to document and explain how the competition unfolded rather than discussing broader issues surrounding the teams or sport.",

    "Off-Field News": "A sports-related news article that focuses on events occurring outside of actual competition. Topics may include player transfers, injuries, coaching changes, contracts, league policies, business operations, legal matters, disciplinary actions, ownership, facilities, or other organizational developments. The article reports news connected to sports without primarily describing the action or results of a game.",

    "Article": "A standard news article whose primary purpose is to report factual information about an event, development, announcement, or situation in a clear, objective, and balanced manner. The article presents verified facts, statements from relevant sources, and essential context without emphasizing urgency, in-depth storytelling, personal opinion, or extensive interpretation. Unlike breaking news, it is not defined by immediacy; unlike a feature, it does not focus on narrative or human-interest storytelling; and unlike analysis, it does not primarily interpret the significance or implications of events. Its main goal is to accurately inform readers about what happened."
}

# print(MAPPING)

MAPPED = {v: k for k , v in MAPPING.items()}

# selected model for intent selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_article_format(article):
    """
    classifies format of article using above selected model / format
    returns dict of format / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=FORMATS,
        hypothesis_template="This article is formatted as a {}",
        multi_label=False
    )

    # sort into format list for use in sql
    format = []

    for rank in range(len(result['labels'])):
        format.append({
            'format': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return format

