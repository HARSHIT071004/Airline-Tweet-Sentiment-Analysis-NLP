import streamlit as st
import joblib
import re

# ----------------------------
# Load Best Model & TF-IDF
# ----------------------------
model_path = 'outputs/best_model_logistic_regression.pkl'  # update based on your training output
model = joblib.load(model_path)

tfidf_path = 'outputs/tfidf_vectorizer.pkl'
tfidf = joblib.load(tfidf_path)

sentiment_map = {0:'Negative', 1:'Neutral', 2:'Positive'}

CUSTOM_STOPWORDS = {"the","a","an","is","in","on","at","to","for","and","or","but","this","that","it","i","you","we","he","she","they"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]
    return ' '.join(tokens)

st.set_page_config(page_title="Airline Tweet Sentiment Analysis", page_icon="✈️", layout="centered")

st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("Type a tweet to predict its sentiment:")

tweet = st.text_area("Enter Tweet", height=150)

if st.button("Predict Sentiment"):
    if tweet.strip():
        clean_tweet = preprocess_text(tweet)
        features = tfidf.transform([clean_tweet])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Sentiment: **{sentiment_map[prediction]}**")
    else:
        st.warning("Please enter some text to predict sentiment.")

st.subheader("Sentiment Distribution (from training data)")
st.image('outputs/sentiment_distribution.png', caption='Sentiment Distribution')

best_model_name = model_path.split('best_model_')[-1].replace('.pkl','')
cm_path = f"outputs/confusion_matrix_{best_model_name}.png"
st.subheader("Confusion Matrix of Best Model")
st.image(cm_path, caption=f"Confusion Matrix - {best_model_name}")
