# pipeline/sentiment.py
# code for full article sentiment analysis (adding entity level later)

from transformers import pipeline

# load model
full_sentiment = pipeline(
    "sentiment-analysis",
    model='cardiffnlp/twitter-roberta-base-sentiment-latest'
)

ent_sentiment = pipeline(
    "text-classification", 
    model="yangheng/deberta-v3-base-absa-v1.1"
)



def full_article_sent(doc):
    """
    function to determine the sentiment of the entire article
    """
    scores=[]
    pos=0
    neg=0
    neu=0

    sent_sent = []

    # Batch all sentences into one call instead of one model call per
    # sentence - some articles run 500+ sentences, and per-call overhead
    # (tokenization/padding setup) dominates at that scale. Same model, same
    # inputs, same per-sentence result - just grouped into fewer forward
    # passes.
    sentences = list(doc.sents)
    results = full_sentiment([s.text for s in sentences]) if sentences else []

    for sent, result in zip(sentences, results):
        sent_sent.append({
            "sentence": sent.text,
            "sentiment": result['label'],
            "score": result['score']
        })
        if result['label'] == "positive":
            score = result['score']
            pos+=1
        elif result['label'] == "negative":
            score = -result['score']
            neg+=1
        else:
            score=0
            neu+=1

        scores.append(score)

    sentiment = sum(scores)/len(scores) if scores else 0.0

    article_sent = {
        'score': sentiment,
        'sentence count': len(scores),
        'positive sentences': pos,
        'negative sentences': neg,
        'neutral sentences': neu
        }
    
    return(article_sent, sent_sent)

def ent_sent(entities, doc):
    """
    entity level sentiment analysis
    """

    sent_list = list(doc.sents)

    # Flatten every (entity, sentence) pair across all entities into one
    # batched call instead of a call per pair - an article with dozens of
    # entities each mentioned several times was previously one model call per
    # mention. text_pair batches as a list of {"text", "text_pair"} dicts.
    pairs = []
    inputs = []
    for text, info in entities.items():
        for sent_id in info["sentences"]:
            pairs.append(text)
            inputs.append({"text": sent_list[sent_id].text, "text_pair": text})

    results = ent_sentiment(inputs) if inputs else []

    scores_by_entity = {}
    for text, result in zip(pairs, results):
        if result['label'] == "Positive":
            score = result['score']
        elif result['label'] == "Negative":
            score = -result['score']
        else:
            score = 0
        scores_by_entity.setdefault(text, []).append(score)

    return {
        text: (sum(scores) / len(scores), len(scores))
        for text, scores in scores_by_entity.items()
    }



# result_food = ent_sentiment("The food was great, but I hated the waiter, and I know Bob hated the food", text_pair='food')
# print(result_food)
# WHEN ON WIFI DOUBLECHECK OUTPUT SCHEMA DOCS R WEIRD

