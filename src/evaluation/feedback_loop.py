import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.model_development.hybrid_model import HybridModel  # Import your model here
from src.data_preprocessing.data_loader import load_data


def evaluate_feedback(feedback_data_path):
    try:
        # Load feedback data with error handling
        feedback_data = pd.read_csv(
            feedback_data_path, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)

        # Ensure the necessary columns are present
        required_columns = {'actual', 'predicted'}
        if not required_columns.issubset(feedback_data.columns):
            raise ValueError(f"CSV file must contain the following columns: {
                             required_columns}")

        y_true = feedback_data['actual']
        y_pred = feedback_data['predicted']

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except pd.errors.ParserError as pe:
        print(f"ParserError: {pe}")
    except Exception as e:
        print(f"An error occurred: {e}")


def update_model(feedback_data_path):
    try:
        # Load feedback data with error handling
        feedback_data = pd.read_csv(
            feedback_data_path, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)

        # Ensure the necessary columns are present
        required_columns = {'actual', 'predicted'}
        if not required_columns.issubset(feedback_data.columns):
            raise ValueError(f"CSV file must contain the following columns: {
                             required_columns}")

        # Extract features and labels for retraining
        X_feedback = feedback_data.drop(columns=['actual', 'predicted'])
        y_feedback = feedback_data['actual']

        # Initialize model
        model = HybridModel()

        # Retrain model with feedback data
        model.train(X_feedback, y_feedback)

        # Save the updated model
        model.save('updated_recommendation_model.pkl')
        print("Model updated with feedback data.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except pd.errors.ParserError as pe:
        print(f"ParserError: {pe}")
    except Exception as e:
        print(f"An error occurred: {e}")
