import pandas as pd
import numpy as np
import streamlit as st
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Cache download of NLTK resources
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk()

# Load model and vectorizer
model = joblib.load('model.pkl')
tf_idf_v = joblib.load('tfidf.pkl')

# Setup stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Sentiment prediction function
def predict_sentiment(x):
    clean_text = re.sub("<.*?>", "", x)
    clean_text = re.sub(r'[^\w\s]', "", clean_text)
    clean_text = clean_text.lower()
    tokenized_text = word_tokenize(clean_text)
    filtered_text = [word for word in tokenized_text if word not in stop_words]
    stemmed_text = [stemmer.stem(word) for word in filtered_text]
    tfidf_review = tf_idf_v.transform([' '.join(stemmed_text)])
    sentiment_prob = model.predict_proba(tfidf_review)[0][1]
    return "positive" if sentiment_prob > 0.6 else "negative"

# Streamlit UI
st.title('Sentiment Analysis')
review_predict = st.text_area('Enter your review:')
if st.button('Predict Sentiment'):
    prediction = predict_sentiment(review_predict)
    st.write("Predicted Sentiment:", prediction)
