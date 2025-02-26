# sentiment.py
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Ensure required NLTK data is downloaded
nltk.download('punkt')

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

def get_anc_sentiment(text):
    """
    Tokenize text into sentences, filter for those mentioning noise cancellation,
    and compute an average sentiment score (1-5 stars).
    """
    sentences = sent_tokenize(text)
    # Filter sentences that mention "noise cancellation" or "anc"
    anc_sentences = [s for s in sentences if "noise cancellation" in s.lower() or "anc" in s.lower()]
    if not anc_sentences:
        return None
    scores = []
    for s in anc_sentences:
        result = sentiment_pipeline(s)
        label = result[0]['label']  # e.g., "5 stars"
        try:
            stars = int(label.split()[0])
        except Exception:
            stars = 3  # default to neutral
        scores.append(stars)
    avg_score = sum(scores) / len(scores)
    return avg_score
