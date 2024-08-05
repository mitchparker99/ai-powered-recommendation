# feature_engineering.py
import pandas as pd


def generate_features(data):
    # Example feature engineering
    data['text_length'] = data['Review Text'].apply(lambda x: len(x))
    return data
