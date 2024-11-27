
import os
import re
import nltk
nltk.download('punkt_tab')
import gensim
import joblib
import numpy as np
import pandas as pd 
import streamlit as st
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For progress bar

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Define paths to model files (assuming they're in the same directory as app.py)
WORD2VEC_MODEL_PATH = "word2vec_model.joblib"
XGB_MODEL_PATH = "best_xgb_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
SCALER_PATH = "scaler.joblib"



# Load pre-trained models and scaler
try:
    word2vec_model = joblib.load(WORD2VEC_MODEL_PATH)
    model = joblib.load(XGB_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    class_names = encoder.classes_
    vector_size = word2vec_model.vector_size
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
   


def clean_text(text):
    if not isinstance(text, str):      
        return ""
    text = text.lower()                 
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'@\w+', '', text)    
    text = re.sub(r'#\w+', '', text)    
    text = re.sub(r'\d+', '', text)     
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = word_tokenize(text)        
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(cleaned_tokens) 



def get_avg_word2vec(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(vector_size) 
    return np.mean([model.wv[token] for token in valid_tokens], axis = 0)



# Streamlit UI
st.title('Ham / Spam Detection')
st.write('Enter a ham or spam content, and the model will classify it.')

# User input
message_text = st.text_area('Enter the message here:')

# Predicting the sentiment
if message_text:
    with st.spinner('Processing...'):  
        # Clean the review text
        cleaned_message = clean_text(message_text)
        st.write(f"**Cleaned Review:** {cleaned_message}")

        # Tokenize and get Word2Vec embedding
        tokens = cleaned_message.split()
        message_embedding = get_avg_word2vec(tokens, word2vec_model, vector_size)

        # Scale the embedding
        message_embedding_scaled = scaler.transform([message_embedding])

        # Predict sentiment
        prediction = model.predict(message_embedding_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0]

        # Map the label to sentiment (Positive/Negative)
        sentiment = "Positive" if predicted_label == 1 else "Negative"

        # Display results
        st.write(f"The predicted label is: **{predicted_label}** ({sentiment})")

        # Display Class Mapping
        st.write("### Class Mapping:")
        st.table({i: label for i, label in enumerate(class_names)})
