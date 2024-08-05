# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill missing values
data['Title'].fillna('', inplace=True)
data['Review_Text'].fillna('', inplace=True)

# Convert categorical columns to numerical
label_encoder = LabelEncoder()
data['Division Name'] = label_encoder.fit_transform(data['Division Name'])
data['Department Name'] = label_encoder.fit_transform(data['Department Name'])
data['Class Name'] = label_encoder.fit_transform(data['Class Name'])

# Features and target
X = data[['Title', 'Review_Text']]
y = data['Class Name']

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_text = tfidf_vectorizer.fit_transform(X['Review_Text'])

# Combine text and other features
X_combined = pd.concat([pd.DataFrame(X_text.toarray()),
                       data[['Division Name', 'Department Name']]], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
