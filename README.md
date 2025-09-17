Airline Tweet Sentiment Analysis (NLP)
Overview

This project performs sentiment analysis on airline-related tweets using Natural Language Processing (NLP). The objective is to classify tweets as Positive, Negative, or Neutral, providing insights into public opinion about different airlines.

Dataset

Source: Tweets.csv

Features include:

Text: The content of the tweet

Airline: The airline the tweet refers to

Sentiment: Label indicating sentiment (Positive/Negative/Neutral)

Additional metadata: Date, Tweet ID, and other relevant fields

Preprocessing and Feature Engineering

Removed punctuation, special characters, and stopwords

Converted all text to lowercase for normalization

Tokenized text and applied TF-IDF vectorization

Handled class imbalance if necessary

Split data into training and testing sets

Modeling Approach

Built and compared multiple classification models, e.g.:

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

Tuned hyperparameters using GridSearchCV

Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices

Tech Stack

Language: Python 3.x

Libraries: pandas, numpy, scikit-learn, nltk or spaCy, matplotlib, seaborn, streamlit (for deployment)

How to Use
1. Clone the Repository
git clone https://github.com/HARSHIT071004/Airline-Tweet-Sentiment-Analysis-NLP.git
cd Airline-Tweet-Sentiment-Analysis-NLP

2. Install Dependencies
pip install -r requirements.txt

3. Train and Evaluate Models

Open the Jupyter Notebook (NLP_Modelling.ipynb)

Or run the training script to preprocess data and build models

4. Run the Web App (Optional)
streamlit run app.py


Enter a tweet in the input box and get the predicted sentiment instantly.

Results

Achieved strong classification performance with high accuracy.

Confusion matrix and evaluation metrics indicate balanced performance across classes.

Random Forest or SVM (depending on your experiment results) delivered the best accuracy.

Future Enhancements

Integrate Word2Vec, GloVe, or BERT embeddings for better feature representation.

Improve performance on Neutral class predictions.

Deploy a live web app for real-time sentiment monitoring.

Expand analysis to multiple social media platforms.

Author

Harshit Sharma

GitHub: HARSHIT071004
