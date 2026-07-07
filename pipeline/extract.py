import trafilatura

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

    title = ""

    if metadata and metadata.title:
        title = metadata.title

    # returns as dict
    return {
        "title": title,
        "text": text or "",
        "url": url,
        "word count": len(text.split())
    }