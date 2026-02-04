import streamlit as st
import pickle

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #2874F0;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
}

.card {
    background-color: #f9f9f9;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.result-positive {
    background-color: #e8f7ee;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
}

.result-negative {
    background-color: #fde8e8;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("<div class='main-title'>🛍️ Flipkart Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze customer emotions from product reviews using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.title("ℹ️ About Project")
st.sidebar.info(
    """
    **Flipkart Review Sentiment Analysis**

    - Algorithm: Logistic Regression  
    - Vectorizer: TF-IDF  
    - Task: Binary Sentiment Classification  
    - Output: Positive / Negative
    """
)

# ------------------ Load Model ------------------
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# ------------------ Input Card ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

review_text = st.text_area(
    "✍️ Paste a customer review below",
    placeholder="Example: The product quality exceeded my expectations and delivery was quick.",
    height=160
)

predict_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Prediction ------------------
if predict_btn:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        with st.spinner("Processing review..."):
            transformed_text = vectorizer.transform([review_text])
            result = model.predict(transformed_text)

        st.markdown("<br>", unsafe_allow_html=True)

        if result[0] == 1:
            st.markdown(
                "<div class='result-positive'>✅ Positive Sentiment Detected 😄</div>",
                unsafe_allow_html=True
            )
            st.caption("Customers are likely satisfied with this product.")
        else:
            st.markdown(
                "<div class='result-negative'>❌ Negative Sentiment Detected 😟</div>",
                unsafe_allow_html=True
            )
            st.caption("This review indicates dissatisfaction or issues.")

# ------------------ Footer ------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "End-to-End ML Project • Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
