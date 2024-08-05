# main.py
from src.data_preprocessing.data_loader import load_data, preprocess_data
from src.model_development.hybrid_model import HybridModel
from src.evaluation.feedback_loop import evaluate_feedback, update_model
import pandas as pd


def main():
    # Load and preprocess data
    data = load_data('data/reviews.csv')
    preprocessed_data = preprocess_data(data)

    # Check for duplicates in the columns that are supposed to be unique
    duplicate_entries = preprocessed_data[preprocessed_data.duplicated(
        subset=['Age', 'Clothing ID'], keep=False)]

    if not duplicate_entries.empty:
        print("Duplicate entries found:")
        print(duplicate_entries)

    # Remove duplicates
    preprocessed_data = preprocessed_data.drop_duplicates(
        subset=['Age', 'Clothing ID'])

    # Pivot data to create user-item matrix
    user_item_matrix = preprocessed_data.pivot(
        index='Age', columns='Clothing ID', values='Rating')

    # Handle missing values (NaNs)
    # Option 1: Fill NaNs with a default value (e.g., 0)
    user_item_matrix = user_item_matrix.fillna(0)

    # Option 2: Drop rows or columns with NaNs (less recommended if it results in significant data loss)
    # user_item_matrix = user_item_matrix.dropna(axis=0)  # Drop rows with NaNs
    # user_item_matrix = user_item_matrix.dropna(axis=1)  # Drop columns with NaNs

    # Initialize and train model
    item_descriptions = preprocessed_data['Review Text']
    model = HybridModel()
    model.train(user_item_matrix, item_descriptions)

    # Save model
    model.save('recommendation_model.pkl')

    # Evaluate feedback and update model
    evaluate_feedback('data/feedback_data.csv')
    update_model('data/feedback_data.csv')


if __name__ == "__main__":
    main()
