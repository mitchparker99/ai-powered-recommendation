# **AI-Powered Recommendation System**

## Overview

This project implements an AI-powered recommendation system using machine learning techniques. The model is designed to recommend products based on their descriptions, leveraging TF-IDF vectorization and a RandomForestClassifier.

## Features

Data Cleaning: Preprocess raw data to prepare it for modeling.
Model Training: Train a recommendation model using TF-IDF vectorization and RandomForestClassifier.
Prediction: Make recommendations based on new product descriptions.

## Project Structure

data/: Contains dataset and processed files.
dataset.csv: Raw data file.
processed/cleaned_dataset.csv: Cleaned dataset ready for modeling.
processed/recommendation_model.pkl: Trained recommendation model.
processed/tfidf_vectorizer.pkl: TF-IDF vectorizer used for feature extraction.
src/: Contains Python scripts for data processing and model operations.
clean_data.py: Script to clean and preprocess the dataset.
recommendation_model.py: Script to train the recommendation model.
make_recommendations.py: Script to make predictions using the trained model.

## Installation

### Clone the Repository

git clone https://github.com/yourusername/ai-powered-recommendation.git
cd ai-powered-recommendation

## Set Up the Environment
### Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies
### Install the required Python packages:


```
pip install -r requirements.txt
Data Preparation
```

###Download and Place the Dataset
###Place your raw dataset file (dataset.csv) in the data/ directory.
###Run the Data Cleaning Script & Clean the raw data and save it as cleaned_dataset.csv:

```
python src/clean_data.py
```

## Model Training

### Train the Recommendation Model

```
python src/recommendation_model.py
```
This will generate recommendation_model.pkl and tfidf_vectorizer.pkl in the data/processed/ directory.

## Making Predictions

### Make Predictions with the Trained Model
Use the following script to make predictions based on a new product description:

```
python src/make_recommendations.py
```
Modify the example_description variable in the script to test different descriptions.

If you want to predict the recommendation for the description "HAND WARMER UNION JACK":

```
example_description = "HAND WARMER UNION JACK"
recommendation = make_recommendation(example_description)
print(f'Recommendation for "{example_description}": {recommendation}')
```

## Troubleshooting

FileNotFoundError: Ensure the paths to the dataset and processed files are correct.
ParserError: Check if the dataset has been properly cleaned and formatted.
UnicodeDecodeError: Ensure the correct encoding is used for reading the dataset (e.g., ISO-8859-1).

## Contact

### For any questions or inquiries, please contact:

#### Name: Mitchell Parker
#### Email: mitchelljamesparker99@gmail.com
#### LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/mitchparker99/)
