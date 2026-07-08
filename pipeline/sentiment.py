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

    for sent in doc.sents:
        # print(f'Sentence analyzed:{sent}')
        result = full_sentiment(sent.text)
        sent_sent.append({
            "sentence": sent.text,
            "sentiment": result[0]['label'],
            "score": result[0]['score']
        })
        # print(result)
        if result[0]['label'] == "positive":
            score = result[0]['score']
            pos+=1
        elif result[0]['label'] == "negative":
            score = -result[0]['score']
            neg+=1
        else:
            score=0
            neu+=1
        
        scores.append(score)

    sentiment = sum(scores)/len(scores)

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
    
    entity_scores={}
    sent_list = list(doc.sents)
    pos = 0
    neg = 0
    neu = 0
    for text, info in entities.items():
        scores = []
        for sent_id in info["sentences"]:
            # print(f'Analyzing Sentence:{sent_list[sent_id].text}')
            result = ent_sentiment(sent_list[sent_id].text, text_pair = text)
            if result[0]['label'] == "Positive":
                score = result[0]['score']
                pos+=1
            elif result[0]['label'] == "Negative":
                score = -result[0]['score']
                neg+=1
            else:
                score=0
                neu+=1
            scores.append(score)

        ent_score = sum(scores)/len(scores)
        entity_scores[text] = (ent_score, len(scores))

    return entity_scores



# result_food = ent_sentiment("The food was great, but I hated the waiter, and I know Bob hated the food", text_pair='food')
# print(result_food)
# WHEN ON WIFI DOUBLECHECK OUTPUT SCHEMA DOCS R WEIRD

