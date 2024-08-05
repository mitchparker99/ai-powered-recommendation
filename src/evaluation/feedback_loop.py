# feedback_loop.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.model_development.hybrid_model import HybridModel  # Import your model here
from src.data_preprocessing.data_loader import load_data


def evaluate_feedback(feedback_data_path):
    # Load feedback data
    feedback_data = pd.read_csv(feedback_data_path)
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


def update_model(feedback_data_path):
    # Load feedback data
    feedback_data = pd.read_csv(feedback_data_path)

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
