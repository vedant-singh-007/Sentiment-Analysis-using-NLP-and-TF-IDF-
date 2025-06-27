import pandas as pd
import numpy as np
import streamlit as st
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Download required NLTK resources (only once)
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk()

# ✅ Load the model (includes TfidfVectorizer + classifier pipeline)
model = joblib.load('model.pkl')

# ✅ Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ✅ Sentiment prediction function
def predict_sentiment(x):
    # Clean the text
    clean_text = re.sub("<.*?>", "", x)
    clean_text = re.sub(r'[^\w\s]', "", clean_text)
    clean_text = clean_text.lower()

    # Tokenize, remove stopwords, and stem
    tokenized_text = word_tokenize(clean_text)
    filtered_text = [word for word in tokenized_text if word not in stop_words]
    stemmed_text = [stemmer.stem(word) for word in filtered_text]

    # Join and predict
    cleaned_input = ' '.join(stemmed_text)
    sentiment_prob = model.predict_proba([cleaned_input])[0][1]

    return "positive" if sentiment_prob > 0.6 else "negative"

# ✅ Streamlit UI
st.title('Sentiment Analysis')
review_predict = st.text_area('Enter your review:')

if st.button('Predict Sentiment'):
    if review_predict.strip():
        prediction = predict_sentiment(review_predict)
        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter a review to analyze.")
