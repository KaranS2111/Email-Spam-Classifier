import streamlit as st
import joblib
import pandas as pd

model = joblib.load('Email_Classifier_NB_model.pkl')

def predict_spam(text):
    prediction = model.predict([text])
    return prediction

def main():
    st.title('Email Spam Classifier')
    st.write('Enter the text of an email to determine if it is spam or not.')

    user_input = st.text_area('Enter email text here:', '')

    if st.button('Classify'):
        prediction = predict_spam(user_input)
        if prediction == 'spam':
            st.error('This email is classified as SPAM.')
        else:
            st.success('This email is classified as NOT spam.')

if __name__ == '__main__':
    main()
