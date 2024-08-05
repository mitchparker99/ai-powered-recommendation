# content_based_filtering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedFilteringModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.cosine_similarities = None

    def fit(self, item_descriptions):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(item_descriptions)
        self.cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    def recommend(self, item_id, top_n=10):
        sim_scores = list(enumerate(self.cosine_similarities[item_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        item_indices = [i[0] for i in sim_scores]
        return item_indices


def load_item_descriptions(file_path):
    return pd.read_csv(file_path)['description']
