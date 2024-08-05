import pandas as pd

# Read the dataset
try:
    data = pd.read_csv('data/raw/dataset.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    data = pd.read_csv('data/raw/dataset.csv', encoding='utf-8')

# Example of data cleaning
data['InvoiceDate'] = pd.to_datetime(
    # Convert InvoiceDate to datetime
    data['InvoiceDate'], format='%m/%d/%Y %H:%M')
# Drop rows where Description is NaN
data = data.dropna(subset=['Description'])
data['Total'] = data['Quantity'] * data['UnitPrice']  # Create a Total column

# Save cleaned dataset
data.to_csv('data/processed/cleaned_dataset.csv', index=False)
