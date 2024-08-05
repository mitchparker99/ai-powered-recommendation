import pandas as pd
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load('data/processed/recommendation_model.pkl')
tfidf_vectorizer = joblib.load('data/processed/tfidf_vectorizer.pkl')


def make_recommendation(description):
    """
    Given a product description, predict the target using the trained model.
    """
    # Transform the description using TF-IDF vectorizer
    description_tfidf = tfidf_vectorizer.transform([description])

    # Predict the target
    prediction = model.predict(description_tfidf)

    return prediction[0]


# Example usage
if __name__ == "__main__":
    # Example description for prediction
    example_description = "MINI JIGSAW SPACEBOY"

    # Make prediction
    recommendation = make_recommendation(example_description)
    print(f'Recommendation for "{example_description}": {recommendation}')
