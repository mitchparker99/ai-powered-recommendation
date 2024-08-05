# main.py
from src.data_preprocessing.data_loader import load_data, preprocess_data
from src.model_development.hybrid_model import HybridModel
from src.evaluation.feedback_loop import evaluate_feedback, update_model


def main():
    # Load and preprocess data
    data = load_data('data/reviews.csv')
    preprocessed_data = preprocess_data(data)

    # Initialize and train model
    # Example, adjust as necessary
    item_descriptions = preprocessed_data['Review Text']
    user_item_matrix = preprocessed_data.pivot(
        index='Age', columns='Clothing ID', values='Rating')

    model = HybridModel()
    model.train(user_item_matrix, item_descriptions)

    # Save model
    model.save('recommendation_model.pkl')

    # Evaluate feedback and update model
    evaluate_feedback('feedback_data.csv')
    update_model('feedback_data.csv')


if __name__ == "__main__":
    main()
