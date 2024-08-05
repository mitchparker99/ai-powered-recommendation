import pandas as pd

# Load the dataset
data = pd.read_csv('data/dataset.csv')

# Data cleaning and preprocessing
data['Review_Text'].fillna('', inplace=True)  # Fill missing review texts
data['Recommended IND'] = data['Recommended IND'].astype(int)  # Ensure target is integer

# Save cleaned dataset
data.to_csv('data/processed/cleaned_dataset.csv', index=False)
