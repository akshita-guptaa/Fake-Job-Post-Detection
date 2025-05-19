import streamlit as st
import joblib
import numpy as np

# Load the trained model and TF-IDF vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit App UI
st.title("🕵️‍♂️ Fake Job Post Detection")
st.subheader("Enter Job Posting Details Below")

# Input fields for the job description and requirements
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")

if st.button("Predict"):
    if description.strip() == "" or requirements.strip() == "":
        st.warning("Please fill out both description and requirements.")
    else:
        # Combine inputs
        combined_text = description + " " + requirements
        
        # Preprocess and vectorize input
        text_vector = vectorizer.transform([combined_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ This job posting is likely **FAKE**.")
        else:
            st.success("✅ This job posting appears to be **REAL**.")

        # Show probabilities
        st.markdown("#### Prediction Probabilities")
        st.write(f"Real: {prediction_proba[0]*100:.2f}%")
        st.write(f"Fake: {prediction_proba[1]*100:.2f}%")
