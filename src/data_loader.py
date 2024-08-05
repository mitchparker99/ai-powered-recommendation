import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Handle missing values
    return df

def split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df
