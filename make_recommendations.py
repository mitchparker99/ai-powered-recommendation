# make_recommendations.py

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
model = joblib.load('recommendation_model.pkl')

# Load and preprocess new data
new_data = pd.DataFrame({
    'Title': ['New Item Title'],
    'Review_Text': ['This is a review text for the new item.']
})

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_text = tfidf_vectorizer.fit_transform(new_data['Review_Text'])

# Combine text features
X_combined = pd.DataFrame(X_text.toarray())

# Make predictions
predictions = model.predict(X_combined)
print(f'Predicted Class: {predictions[0]}')
