# data_loader.py
import pandas as pd


def load_data(file_path):
    try:
        # Handle parsing errors
        return pd.read_csv(file_path, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        raise


def preprocess_data(data):
    # Perform any necessary preprocessing (e.g., handling missing values)
    data.fillna('', inplace=True)
    return data
