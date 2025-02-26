from .sentiment import get_anc_sentiment

def extract_product_info(text):
    """
    Extract product type and ANC quality from text.
    """
    text_lower = text.lower()
    # Determine product type (can be extended to more categories)
    if "earbud" in text_lower or "earphone" in text_lower:
        product_type = "earphone"
    elif "headphone" in text_lower:
        product_type = "headphone"
    else:
        product_type = "unknown"
    
    # Use the sentiment function to evaluate noise cancellation quality
    anc_sent = get_anc_sentiment(text)
    if anc_sent is not None:
        if anc_sent >= 4:
            anc_quality = "strong"
        elif anc_sent >= 3:
            anc_quality = "moderate"
        else:
            anc_quality = "none"
    else:
        anc_quality = "none"
    
    return product_type, anc_quality
