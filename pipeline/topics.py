from transformers import pipeline


# pre-set topic names for use in classification - to be edited
TOPIC_NAMES = [
    "Politics, government, elections, legislation, public policy, and political campaigns",

    "Business, companies, markets, finance, investing, economics, and corporate news",

    "Technology, artificial intelligence, software, hardware, science, and innovation",

    "Sports, athletes, teams, competitions, tournaments, and sporting events",

    "Entertainment, celebrities, movies, television, music, and popular culture",

    "Geopolitics, international relations, diplomacy, conflicts, and foreign affairs",

    "Lifestyle, wellness, travel, food, health, and leisure activities",

    "Religion, faith, religious leaders, beliefs, and spiritual communities"
]

MAPPING = {
    "Politics": "Politics, government, elections, legislation, public policy, and political campaigns",

    "Business": "Business, companies, markets, finance, investing, economics, and corporate news",

    "Technology" : "Technology, artificial intelligence, software, hardware, science, and innovation",

    "Sports": "Sports, athletes, teams, competitions, tournaments, and sporting events",

    "Entertainment": "Entertainment, celebrities, movies, television, music, and popular culture",

    "Geoploitics": "Geopolitics, international relations, diplomacy, conflicts, and foreign affairs",

    "Lifestyle": "Lifestyle, wellness, travel, food, health, and leisure activities",

    "Religion": "Religion, faith, religious leaders, beliefs, and spiritual communities"
}

MAPPED = {v: k for k , v in MAPPING.items()}

# selected model for topic selection
model = pipeline(
    "zero-shot-classification",
    model='facebook/bart-large-mnli'
)

def classify_article_topics(article):
    """
    classifies topic of article using above selected model / topics
    returns dict of topics / ranks, and confidence levels
    """
    # create var classification tet ot assign what text will be used
    classification_text=article['text'][:1000]

    # get raw output in results
    result = model(
        classification_text,
        candidate_labels=TOPIC_NAMES,
        hypothesis_template="The main topic of this news is {}",
        multi_label=True
    )

    # sort into topics list for use in sql
    topics = []

    for rank in range(len(result['labels'])):
        topics.append({
            'topic': MAPPED[result['labels'][rank]],
            'confidence': float(result['scores'][rank]),
            'rank': rank+1
        })

    return topics

