from sklearn.feature_extraction.text import TfidfVectorizer
import re


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    return text

