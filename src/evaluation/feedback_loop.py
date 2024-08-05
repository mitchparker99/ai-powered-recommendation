import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.model_development.hybrid_model import HybridModel


def evaluate_feedback(feedback_data_path):
    try:
        # Load feedback data
        feedback_data = pd.read_csv(
            feedback_data_path, encoding='utf-8', on_bad_lines='warn')

        # Print the columns of the DataFrame for debugging
        print("Columns in the CSV file:", feedback_data.columns.tolist())

        # Example evaluation using 'Rating'
        y_true = feedback_data['Rating']

        # Replace with actual predictions if available
        y_pred = feedback_data['Rating']

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
        # Load feedback data
        feedback_data = pd.read_csv(
            feedback_data_path, encoding='utf-8', on_bad_lines='warn')

        # Print the columns and data types for debugging
        print("Columns in the CSV file:", feedback_data.columns.tolist())
        print("Data types:", feedback_data.dtypes)

        # Drop rows with missing values in columns we use for training
        feedback_data = feedback_data.dropna(subset=['Rating'])

        # Convert categorical columns to numeric
        categorical_cols = ['Title', 'Review_Text',
                            'Division_Name', 'Department_Name', 'Class_Name']
        for col in categorical_cols:
            # Print unique values in the column
            print(f"Processing column: {col}")
            print(feedback_data[col].unique())

            # Ensure the column is of type object (string)
            if feedback_data[col].dtype == 'object':
                le = LabelEncoder()
                feedback_data[col] = le.fit_transform(
                    feedback_data[col].astype(str))
            else:
                print(f"Skipping column {col} as it is not of type 'object'.")

        # Extract features and labels
        X_feedback = feedback_data.drop(columns=['Rating'])
        y_feedback = feedback_data['Rating'].astype(float)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_feedback, y_feedback, test_size=0.2, random_state=42)

        # Initialize and train model
        model = HybridModel()
        model.train(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Save the updated model
        model.save('updated_recommendation_model.pkl')
        print("Model updated with feedback data.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except pd.errors.ParserError as pe:
        print(f"ParserError: {pe}")
    except Exception as e:
        print(f"An error occurred: {e}")
