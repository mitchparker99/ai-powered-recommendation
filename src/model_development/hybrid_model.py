def hybrid_model(user_vector, product_vectors, collaborative_matrix):
    content_sim = content_based_recommendations(user_vector, product_vectors)
    collaborative_sim = np.dot(collaborative_matrix, user_vector.T)
    return 0.5 * content_sim + 0.5 * collaborative_sim
