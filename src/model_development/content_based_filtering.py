from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendations(user_vector, product_vectors):
    similarities = cosine_similarity(user_vector, product_vectors)
    return similarities
