import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEEMER = PorterStemmer()

# Load saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@st.cache_data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data
def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [STEEMER.stem(word) for word in tokens]    
    return " ".join(tokens)

@st.cache_data
def predict_spam(message):
    cleaned = clean_text(message)
    processed = preprocess_text(cleaned)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)[0]
    return "Spam" if pred == 1 else "Not Spam"

st.title('Email Spam Classifier System')
st.write('Enter an email message and click Predict to classify it as Spam or Not Spam.')


user_input = st.text_area("Email Text", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.write("Please enter some text to classify.")
    else:
        label = predict_spam(user_input)
        if label == "Spam":
            st.error('Prediction :Spam')
    
        else:
            st.success('Prediction : Not Spam')




