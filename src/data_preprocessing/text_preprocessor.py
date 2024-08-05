# text_preprocessor.py
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix
