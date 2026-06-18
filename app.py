import streamlit as st
import pickle

# Page Config
st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="📊"
)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📊 AI-Powered Customer Sentiment Analysis")
st.write("Analyze customer reviews and predict sentiment using Machine Learning.")

# Input
review = st.text_area("Enter Customer Review")

# Prediction
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        st.subheader("Prediction Result")

        if prediction.lower() == "positive":
            st.success("😊 Positive Sentiment")

        elif prediction.lower() == "negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")