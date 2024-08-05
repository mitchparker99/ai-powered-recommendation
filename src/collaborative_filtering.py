import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class CollaborativeFilteringModel:
    def __init__(self, n_neighbors=5):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')

    def fit(self, user_item_matrix):
        self.model.fit(user_item_matrix)

    def recommend(self, user_id, user_item_matrix):
        distances, indices = self.model.kneighbors(user_item_matrix[user_id])
        return indices


def load_user_item_matrix(file_path):
    return pd.read_csv(file_path, index_col=0)
