from sklearn.decomposition import TruncatedSVD


def collaborative_filtering(train_data):
    # Matrix factorization with SVD
    model = TruncatedSVD(n_components=20)
    return model.fit_transform(train_data)
