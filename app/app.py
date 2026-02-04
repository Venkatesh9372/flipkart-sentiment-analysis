import streamlit as st
import pickle

# ------------------ Page Config ------------------
st.markdown("""
<style>
.single-line-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    white-space: nowrap;
}
</style>

<h1 class="single-line-title">🛒 Flipkart Review Sentiment Analysis</h1>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Enter a product review and predict whether the sentiment is positive or negative."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------ Input ------------------
user_review = st.text_area(
    "Customer Review",
    placeholder="Example: The product quality is excellent and delivery was fast...",
    height=150
)

st.markdown("""
<style>
div.stButton > button {
    background-color: #2874F0;  /* Blue */
    color: white;
    font-size: 16px;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 16px;
    border: none;
}

div.stButton > button:hover {
    background-color: #1f5ed9; /* Darker blue on hover */
}
</style>
""", unsafe_allow_html=True)

# ------------------ Prediction ------------------
if st.button("🔍 Predict Sentiment", use_container_width=True):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            review_vector = vectorizer.transform([user_review])
            prediction = model.predict(review_vector)

        st.markdown("---")

        if prediction[0] == 1:
            st.success("✅ **Positive Review** 😊")
            st.markdown(
                "<div style='background-color:#e6f4ea; padding:15px; border-radius:8px;'>"
                "This review expresses a positive sentiment toward the product."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.error("❌ **Negative Review** 😞")
            st.markdown(
                "<div style='background-color:#fdecea; padding:15px; border-radius:8px;'>"
                "This review expresses a negative sentiment toward the product."
                "</div>",
                unsafe_allow_html=True
            )

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "Built using TF-IDF and Logistic Regression"
    "</p>",
    unsafe_allow_html=True
)
