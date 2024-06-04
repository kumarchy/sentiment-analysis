import pickle
import streamlit as st
import pandas as pd
import requests
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

new = pickle.load(open('reviews.pkl', 'rb'))
new_df = pd.DataFrame(new)
x = new_df['review']
y = new_df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

svc = LinearSVC(C=1, loss='hinge')
svc.fit(x_train_vectorized, y_train)

svc_pred = svc.predict(x_test_vectorized)
svc_acc = accuracy_score(y_test, svc_pred)

st.header("Sentiment Analysis")
input_text = st.text_input("Enter the review")
if input_text:
    input_vectorized = vectorizer.transform([input_text])

    prediction = svc.predict(input_vectorized)

    show_button = st.button('Show Sentiment')
    if show_button:
        st.subheader("Predicted Sentiment:")
        if prediction[0] == 1:
            st.write("Positive")
        else:
            st.write("Negative")
