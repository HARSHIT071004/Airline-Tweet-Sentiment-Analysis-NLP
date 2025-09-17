Airline Tweet Sentiment Analysis | NLP
Overview

This project applies Natural Language Processing (NLP) to analyze airline-related tweets and classify them as Positive, Negative, or Neutral. It demonstrates a complete pipeline—from preprocessing and feature engineering to machine learning model evaluation—providing insights into public sentiment toward airlines.

Dataset

Source: Tweets.csv

Records: ~14,600 tweets

Features:

Text – Tweet content

Airline – Airline referenced in the tweet

Sentiment – Label (Positive/Negative/Neutral)

Metadata – Date, Tweet ID, and additional attributes

Project Workflow
1. Data Cleaning

Removed punctuation, special characters, and stopwords

Normalized text by converting to lowercase

Handled missing or inconsistent values

2. Exploratory Data Analysis (EDA)

Examined sentiment distribution across airlines

Created count plots, bar charts, and word clouds for sentiment classes

Analyzed relationships between airlines and sentiment trends

3. Feature Engineering

Tokenized text

Applied TF-IDF vectorization for feature extraction

Addressed class imbalance where necessary

Split the dataset into training and testing sets

4. Model Building

Trained multiple classification models:

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

Performed GridSearchCV for hyperparameter tuning

5. Model Evaluation

Evaluated models using Accuracy, Precision, Recall, F1-score, and Confusion Matrices

Compared model performance to select the best-performing classifier

Key Insights

SVM and Logistic Regression provided strong accuracy across classes.

Class imbalance slightly affected Neutral predictions.

Tweets referencing customer service issues drove most negative sentiment.

Technology Stack

Language: Python 3.x

Libraries: pandas, numpy, scikit-learn, nltk or spaCy, matplotlib, seaborn, streamlit

Output Files

Preprocessed data saved as cleaned_tweets.csv

Visualization outputs (word clouds, sentiment distributions) saved as .png images

How to Run

Clone the Repository

git clone https://github.com/HARSHIT071004/Airline-Tweet-Sentiment-Analysis-NLP.git
cd Airline-Tweet-Sentiment-Analysis-NLP


Install Dependencies

pip install -r requirements.txt


Train and Evaluate Models

Open the Jupyter Notebook: NLP_Modelling.ipynb

Or run the training script to preprocess data and build models

Run the Web App (Optional)

streamlit run app.py


Enter a tweet in the input field to instantly get the predicted sentiment

Results

Achieved high classification accuracy across sentiment classes

Confusion matrices confirmed balanced performance

SVM consistently outperformed other models in precision and recall

Future Enhancements

Use advanced embeddings such as Word2Vec, GloVe, or BERT

Improve classification for Neutral tweets

Deploy as a real-time web application for live sentiment monitoring

Expand to multi-platform sentiment analysis

Author

Harshit Sharma

GitHub: HARSHIT071004
