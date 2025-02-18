import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure this is the vectorizer, not the model
model = joblib.load("logistic_regression_model.pkl")  # Ensure this is the model

# Streamlit UI
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("🔍 Enter a job description to check if it's real or fake.")

# User input
job_description = st.text_area("📝 Job Description", height=150)

# Create columns for button alignment
col1, col2 = st.columns([1, 4])  # Adjust the numbers to change the button positioning

with col1:
    check_button = st.button("🔍 Check")

with col2:
    clear_button = st.button("❌ Clear Text")

# Clear text feature
if clear_button:
    job_description = ""  # Reset the job description to an empty string

if check_button:
    if job_description:
        # Transform input using the saved vectorizer
        job_vector = vectorizer.transform([job_description])
        
        # Predict
        prediction = model.predict(job_vector)[0]
        
        # Display result
        if prediction == 1:
            st.error("🚨 Warning: This job posting is likely **FAKE**! 🚨")
        else:
            st.success("✅ This job posting seems **REAL**! 🎯")
    else:
        st.warning("⚠️ Please enter a job description.")
