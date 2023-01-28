import streamlit as st
import pickle

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_message (messages):
    messages = re.sub('[^a-zA-Z0-9]',' ',messages)
    messages=messages.lower()
    messages=messages.split()
    messages=[ps.stem(word)for word in messages if not word in stopwords.words('english')]
    messages=' '.join(messages)
    return messages

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
input_message=st.text_area("Enter the message")


if st.button("predict"):
#preprocess
    transformed_message=transform_message(input_message)
    #vectorisation
    vectorised_input=tfidf.transform([transformed_message])
    #prediction
    result=model.predict(vectorised_input)[0]
    print(result)
    #display
    if result==1:
        st.header("Spam!!!")
    else:
        st.header("Not Spam :)")