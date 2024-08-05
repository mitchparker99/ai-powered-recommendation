# hybrid_model.py
from src.model_development.collaborative_filtering import CollaborativeFilteringModel
from src.model_development.content_based_filtering import ContentBasedFilteringModel


class HybridModel:
    def __init__(self):
        self.cf_model = CollaborativeFilteringModel()
        self.cb_model = ContentBasedFilteringModel()

    def train(self, user_item_matrix, item_descriptions):
        self.cf_model.fit(user_item_matrix)
        self.cb_model.fit(item_descriptions)

    def recommend(self, user_id, item_id, user_item_matrix, top_n=10):
        cf_recommendations = self.cf_model.recommend(user_id, user_item_matrix)
        cb_recommendations = self.cb_model.recommend(item_id, top_n)
        combined_recommendations = list(
            set(cf_recommendations) | set(cb_recommendations))
        return combined_recommendations

    def save(self, file_path):
        # Save the model (pseudo-code, use actual saving methods)
        pass

    def load(self, file_path):
        # Load the model (pseudo-code, use actual loading methods)
        pass
