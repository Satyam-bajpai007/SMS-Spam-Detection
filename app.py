import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
pr = PorterStemmer()

def transforms(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y.copy()
    y.clear()
    for i in text:
        y.append(pr.stem(i))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS")

if st.button("Predict"):
    transform_sms = transforms(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
