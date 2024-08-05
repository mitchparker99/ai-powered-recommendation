import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned data
data = pd.read_csv('data/processed/cleaned_dataset.csv')

# Features and labels
X = data['Description']  # Assuming Description is the feature for TF-IDF
y = data['Country']  # Assuming Country is the target

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Save the model and vectorizer
joblib.dump(model, 'data/processed/recommendation_model.pkl')
joblib.dump(tfidf_vectorizer, 'data/processed/tfidf_vectorizer.pkl')
