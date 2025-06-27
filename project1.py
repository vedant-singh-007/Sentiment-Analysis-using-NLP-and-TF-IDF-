#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk


# In[ ]:


df = pd.read_csv("IMDB_Dataset.csv")


# In[ ]:


df


# In[ ]:


# Clean HTML tags
df['clean_text'] = df['review'].apply(lambda x: re.sub("<.*?>", "", x))


# In[ ]:


# Remove punctuation and special characters
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', "", x))


# In[ ]:


# Convert to lowercase
df['clean_text'] = df['clean_text'].str.lower()


# In[ ]:


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[ ]:


# Tokenize text
df['tokenized_text'] = df['clean_text'].apply(lambda x: word_tokenize(x))


# In[ ]:


# Get English stopwords
stop_words = set(stopwords.words('english'))


# In[ ]:


# Remove stopwords
df['filtered_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])


# In[ ]:


# Initialize Porter Stemmer
porter = PorterStemmer()


# In[ ]:


# Apply stemming
df['lemma_text'] = df['filtered_text'].apply(lambda x: [porter.stem(word) for word in x])


# In[ ]:


# Prepare features and target
X = df['lemma_text']
y = df['sentiment']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()


# In[ ]:


# Transform training data
X_train = tfidf.fit_transform(X_train.apply(lambda x: " ".join(x)))


# In[ ]:


# Transform test data
X_test = tfidf.transform(X_test.apply(lambda x: " ".join(x)))


# In[ ]:


# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# In[ ]:


import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential


# In[ ]:


# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


# Build the model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(2, activation="softmax")  # Changed from sigmoid to softmax for multi-class
])


# In[ ]:


# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Train the model
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))


# In[ ]:


import joblib

# Save the model and vectorizer
model.save('model.keras')  # Use .save() for Keras models
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save label encoder too

