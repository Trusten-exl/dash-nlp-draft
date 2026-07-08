import trafilatura
from langdetect import detect

from bs4 import BeautifulSoup
from typing import Dict, Any
import json

from datetime import datetime


def extract_article_info(url):
    """
    Function to extract article text and title for use in NLP
    """

    # download / check if available
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        return None
    

    # extraxts text
    text = trafilatura.extract(
        downloaded,
    )

    # extracts metadata
    metadata = trafilatura.extract_metadata(
        downloaded
    )

    #tPOssible integration - tables / photos 

    title = getattr(metadata, "title", None)
    hostname = getattr(metadata, "hostname", None)
    sitename = getattr(metadata, "sitename", None)
    description = getattr(metadata, "description", None)
    author = getattr(metadata, "author", None)
    publish_date = normalize_date(getattr(metadata, "date", None))
    language = detect_language(text)

    article = {

        "url": url,

        "canonical_url": None,

        "title": None,

        "description": None,

        "text": "",

        "word_count": 0,

        "hostname": None,

        "sitename": None,

        "author": None,

        "publish_date": None,

        "modified_date": None,

        "language": None,

        "image_url": None,

        "video_url": None
    }

    article = {
        "title": title,
        "description": description,
        "text": text or "",
        "language": language,
        "author": author,
        "url": url,
        "word_count": len(text.split()),
        "hostname": hostname,
        "sitename": sitename,
        "publish_date": publish_date
    }

    html_metadata, json_ld = extract_html_metadata(
        downloaded
    )


    for key, value in html_metadata.items():

        if value and not article.get(key):
            article[key] = value

    for item in json_ld:

        if not isinstance(item, dict):
            continue

        article_type = item.get("@type")

        # Handle cases where @type is a list
        if isinstance(article_type, list):
            is_article = any(
                t in ["NewsArticle", "Article"]
                for t in article_type
            )
        else:
            is_article = article_type in [
                "NewsArticle",
                "Article"
            ]


        if not is_article:
            continue


        # ----------------------
        # Title
        # Trafilatura priority
        # ----------------------

        if not article["title"]:

            article["title"] = item.get(
                "headline"
            )


        # ----------------------
        # Description
        # Trafilatura priority
        # ----------------------

        if not article["description"]:

            article["description"] = item.get(
                "description"
            )


        # ----------------------
        # Author
        # Trafilatura priority
        # ----------------------

        if not article["author"]:

            author = item.get(
                "author"
            )

            if isinstance(author, dict):

                article["author"] = (
                    author.get("name")
                )


            elif isinstance(author, list):

                authors = []

                for person in author:

                    if isinstance(person, dict):

                        name = person.get(
                            "name"
                        )

                        if name:
                            authors.append(
                                name
                            )

                    elif isinstance(person, str):

                        authors.append(
                            person
                        )


                if authors:

                    article["author"] = ", ".join(
                        authors
                    )


            elif isinstance(author, str):

                article["author"] = author



        # ----------------------
        # Publish date
        # Trafilatura priority
        # ----------------------

        if not article["publish_date"]:

            article["publish_date"] = normalize_date(item.get(
                "datePublished"
            ))



        # ----------------------
        # Modified date
        # JSON-LD only source
        # ----------------------

        if not article["modified_date"]:

            article["modified_date"] = normalize_date(item.get(
                "dateModified"
            ))



        # ----------------------
        # Image
        # HTML/OG usually better,
        # JSON-LD fallback
        # ----------------------

        if not article["image_url"]:

            image = item.get(
                "image"
            )

            if isinstance(image, str):

                article["image_url"] = image


            elif isinstance(image, list) and image:

                first_image = image[0]

                if isinstance(first_image, str):

                    article["image_url"] = first_image

                elif isinstance(first_image, dict):

                    article["image_url"] = (
                        first_image.get("url")
                    )


            elif isinstance(image, dict):

                article["image_url"] = (
                    image.get("url")
                )

    # returns as dict
    return article

def detect_language(text: str):
    """
    Detect article language using langdetect.
    """

    if not text:
        return None

    try:
        return detect(text[:5000])

    except Exception:
        return None
    
def extract_json_ld(soup):
    """
    Extract JSON-LD structured metadata.
    """

    results = []

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:

        try:

            data = json.loads(
                script.string
            )

            if isinstance(data, list):
                results.extend(data)

            else:
                results.append(data)

        except Exception:
            continue

    return results



def extract_html_metadata(html):
    """
    Extract metadata from HTML tags.
    """

    soup = BeautifulSoup(
        html,
        "lxml"
    )


    metadata = {

        "image_url": None,
        "video_url": None,
        "locale": None,
        "modified_date": None,
        "keywords": []

    }


    # ----------------------
    # OpenGraph
    # ----------------------

    og_image = soup.find(
        "meta",
        property="og:image"
    )

    if og_image:
        metadata["image_url"] = (
            og_image.get("content")
        )


    og_video = soup.find(
        "meta",
        property="og:video"
    )

    if og_video:
        metadata["video_url"] = (
            og_video.get("content")
        )


    og_locale = soup.find(
        "meta",
        property="og:locale"
    )

    if og_locale:
        metadata["locale"] = (
            og_locale.get("content")
        )


    # ----------------------
    # Twitter fallback image
    # ----------------------

    if not metadata["image_url"]:

        twitter_image = soup.find(
            "meta",
            attrs={
                "name": "twitter:image"
            }
        )

        if twitter_image:

            metadata["image_url"] = (
                twitter_image.get("content")
            )


    # ----------------------
    # Modified date
    # ----------------------

    modified = soup.find(
        "meta",
        property="article:modified_time"
    )

    if modified:

        metadata["modified_date"] = (
            normalize_date(modified.get("content"))
        )


    # ----------------------
    # Keywords
    # ----------------------


    return metadata, extract_json_ld(soup)


def normalize_date(date_value):

    if not date_value:
        return None

    date_value = str(date_value)

    # Handles:
    # 2026-06-23
    # 2026-06-23T15:56:53+0000
    # 2026-06-23T15:56:53Z

    if len(date_value) >= 10:
        return date_value[:10]

    return None

# print(extract_article_info('https://www.cnbc.com/2026/06/23/meta-glasses-are-new-smart-glasses-starting-at-299.html'))

# {'title': 'Meta announces new smart glasses starting at $299, as Zuckerberg keeps pushing wearables', 
#  'description': 'Meta executives have said they see the lightweight smart glasses as a step towards a more advanced device that includes screens in the lenses.', 
#  'text': "Meta on Tuesday announced a new set of $299 smart glasses, at least $80 less than the price tag for the company's entry-level second-generation Meta Ray-Ban glasses, as CEO Mark Zuckerberg continues his push into wearables.\nThe Meta Glasses come with new designs and are built in partnership with Ray-Ban parent EssilorLuxottica, but they don't come with Ray-Ban or Oakley branding.\nMeta is aggressively marketing its smart glasses to consumers as eyewear competition heats up and consumers find more value in augmented reality devices. Though the smart glasses market is still small, Meta and EssilorLuxottica dominate it, with estimated market share of more than 80% and millions of units sold since they first launched in 2021.\nMeta Glasses lack a screen, but they include a camera and personal speakers. Users can speak to Meta's AI to translate or understand what they see around them, or take photos and videos of their surroundings.\nMeta executives have said they see the lightweight smart glasses as a step toward a more advanced device that includes screens in the lenses with computing capabilities. Meta last year announced glasses called Ray-Ban Display glasses, which cost $799 and include a built-in display.\nZuckerberg has found more success in smart glasses than in virtual reality headsets, which were key to the company changing its name from Facebook to Meta in 2021. VR has continued to be a niche market, largely for gamers, but Zuckerberg is focused on owning a hardware platform for the artificial intelligence era.\nMeanwhile, competition in the smart glasses market is picking up. Google said last month that it's building new computerized eyewear in partnership with Warby Parker that will use its Gemini AI model. Last week, Snap announced Specs, a pair of $2,195 smart glasses that CEO Evan Spiegel positioned as the successor to the smartphone.\nMeta Glasses come in three new designs, the company said. Meta also introduced a new charging stand for the glasses.", 
#  'language': 'en', 
#  'author': 'Kif Leswing', 
#  'url': 'https://www.cnbc.com/2026/06/23/meta-glasses-are-new-smart-glasses-starting-at-299.html', 
#  'word count': 323, 
#  'hostname': 'cnbc.com', 
#  'sitename': 'CNBC', 
#  'publish_date': '2026-06-23', 
#  'image_url': 'https://image.cnbcfm.com/api/v1/image/108325134-1782163492984-Meta_Glasses_3_16x9.jpg?v=1782163531&w=1920&h=1080', 
#  'modified_date': '2026-06-23'}