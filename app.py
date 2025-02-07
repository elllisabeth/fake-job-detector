import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("random_search_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("🔍 Enter a job description to check if it's real or fake.")

# User input
job_description = st.text_area("📝 Job Description", height=150)

if st.button("🔍 Check"):
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

# Run with: `streamlit run app.py`
