from src.data_preprocessing.data_loader import load_data, preprocess_data
from src.model_development.hybrid_model import HybridModel
from src.evaluation.feedback_loop import evaluate_feedback, update_model
import pandas as pd


def main():
    # Load and preprocess data
    data = load_data('data/cleaned_feedback_data.csv')
    preprocessed_data = preprocess_data(data)

    # Print column names to verify
    print("Columns in preprocessed data:", preprocessed_data.columns)

    # Rename columns if necessary to match script expectations
    preprocessed_data.rename(columns={
        'Clothing ID': 'Clothing_ID',
        'Review Text': 'Review_Text',
        'Division Name': 'Division_Name',
        'Department Name': 'Department_Name',
        'Class Name': 'Class_Name'
    }, inplace=True)

    # Check for duplicates in the columns that are supposed to be unique
    duplicate_entries = preprocessed_data[preprocessed_data.duplicated(
        subset=['Age', 'Clothing_ID'], keep=False)]

    if not duplicate_entries.empty:
        print("Duplicate entries found:")
        print(duplicate_entries)

    # Remove duplicates
    preprocessed_data = preprocessed_data.drop_duplicates(
        subset=['Age', 'Clothing_ID'])

    # Convert columns to strings where necessary and handle errors
    for column in ['Title', 'Division_Name', 'Department_Name', 'Class_Name']:
        if column in preprocessed_data.columns:
            preprocessed_data[column] = preprocessed_data[column].astype(str)

    # Remove rows that cause errors during processing
    clean_data = pd.DataFrame(columns=preprocessed_data.columns)

    for index, row in preprocessed_data.iterrows():
        try:
            # Example of processing that could fail
            row['Title'].lower()

            # Append valid rows to clean_data using pd.concat
            clean_data = pd.concat([clean_data, pd.DataFrame(
                [row], columns=clean_data.columns)], ignore_index=True)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    if clean_data.empty:
        print("No valid data to process.")
        return

    # Print processed data columns
    print("Processed data columns:", clean_data.columns)

    # Pivot data to create user-item matrix
    if 'Age' in clean_data.columns and 'Clothing_ID' in clean_data.columns and 'Rating' in clean_data.columns:
        user_item_matrix = clean_data.pivot(
            index='Age', columns='Clothing_ID', values='Rating')

        # Handle missing values (NaNs)
        user_item_matrix = user_item_matrix.fillna(0)

        # Generate item descriptions from clean_data
        item_descriptions = clean_data[[
            'Clothing_ID', 'Title', 'Division_Name', 'Department_Name', 'Class_Name']].drop_duplicates()

        # Initialize and train model
        model = HybridModel()
        model.train(user_item_matrix, item_descriptions)

        # Save model
        model.save('recommendation_model.pkl')

    else:
        print("Required columns are missing from the cleaned data.")

    # Evaluate feedback and update model
    evaluate_feedback('data/cleaned_feedback_data.csv')
    update_model('data/cleaned_feedback_data.csv')


if __name__ == "__main__":
    main()
