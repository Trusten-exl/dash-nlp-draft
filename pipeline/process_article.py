from extract import extract_article_info
from topics import classify_article_topics
from article_intent import classify_article_intent
from format import classify_article_format
from sic import classify_sic_article
from save import save_article, save_topics, save_intents, save_formats, save_article_sentiment, save_entities, save_entity_sentiment, save_sic, save_p_orientation, save_p_salience
from sentiment import full_article_sent, ent_sent
from ner import extract_ent
from political import classify_article_salience, classify_poliical_orientation
import spacy

nlp = spacy.load('en_core_web_sm')

def process_article(url):
    """
    Full procesisng pipeline for a single article
    """
    print("Processing Start")

    # extract from url, save article dict
    article = extract_article_info(url=url)

    if article == None:
        print('Article not processed')
        return
    elif article['title'] == None:
        pass
    else:
        print(f'Article: {article['title']}')
        print(article)

    # save to db
    article_id = save_article(article=article)

    print(f'ID: {article_id}')

    # classify topics
    topics = classify_article_topics(article=article)

    # print(f'Topics: {topics}')

    sic_codes = classify_sic_article(article)
    # print(f"\nSIC codes: {sic_codes}")

    if sic_codes["division"][0]["score"] > 0.2:
        # print("\nHIGH SIC SIMILAIRTY - SAVING...")

        save_sic(sic_codes, article_id)

    # save to db
    save_topics(t=topics, article_id=article_id)

    # classify intent
    intent = classify_article_intent(article)

    # print(f'Intent:{intent}')

    # save to db
    save_intents(i=intent, article_id=article_id)

    # classify format
    format = classify_article_format(article=article)

    # print()
    # print(format)

    # save to db
    save_formats(article_id=article_id, f=format)

    # nlp for use in sentence-by-sentence sentiment analysis / ner
    doc=nlp(article['text'])

    # print('NLP complete')

    # print('starting sentiment analysis')
    # full article sentiment
    sentiment = full_article_sent(doc=doc)
    # save to db
    save_article_sentiment(sentiment=sentiment, article_id=article_id)

    # print('Sentiment Analysis Complete')
    # print(sentiment)

    # process article entities
    entities = extract_ent(doc=doc)
    # save to db
    save_entities(entities=entities, article_id=article_id)

    # print('Entities Extracted')

    # calc entity level sent
    entity_sentiment = ent_sent(entities=entities, doc=doc)
    
    # print('Ent Sent Calculated')
    # print(entity_sentiment)
    
    # save to db
    save_entity_sentiment(ent_sent=entity_sentiment, article_id=article_id)


    # politics
    # print('Politics')
    political_orienattion = classify_poliical_orientation(article=article)
    save_p_orientation(political_orienattion, article_id=article_id)
    # print('Orientation: {political_orientation}')

    political_salience = classify_article_salience(article=article)
    save_p_salience(political_salience, article_id)
    # print('Salience: {political_salience}')

    print('Complete')

    return article_id

process_article('https://www.cnbc.com/2026/06/23/meta-glasses-are-new-smart-glasses-starting-at-299.html')
