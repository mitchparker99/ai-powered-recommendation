# data_loader.py
import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    # Perform any necessary preprocessing (e.g., handling missing values)
    data.fillna('', inplace=True)
    return data
