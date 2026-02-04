# 📊 Flipkart Sentiment Analysis (ML)

## 📌 Project Overview
This project focuses on sentiment analysis of real-time Flipkart product reviews using Machine Learning techniques. The goal is to classify customer reviews as **Positive** or **Negative** and deploy the trained model as a web application using **Streamlit**, hosted on **AWS EC2**.

This project was completed as part of my **Data Science Internship with GenAI** at **Innomatics Research Labs**.

## 🎯 Objectives
* Analyze customer reviews to understand sentiment trends
* Preprocess and clean real-world textual data
* Train and evaluate machine learning models for sentiment classification
* Deploy the trained model as a real-time web application
* Host the application on AWS Cloud for public access

## 🧾 Dataset Information
* **Source:** Flipkart (provided by Data Engineering team)
* **Product:** YONEX MAVIS 350 Nylon Shuttle
* **Total Reviews:** 8,518
* **Key Features:**
   * Reviewer Name
   * Review Text
   * Ratings
   * Up Votes / Down Votes
   * Review Date & Location

## ⚙️ Tech Stack Used
* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
* **Text Representation:** TF-IDF
* **Model:** Machine Learning Classifier
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub
* **Cloud Platform:** AWS EC2 (Ubuntu)

## 🧠 Project Workflow
1. Exploratory Data Analysis (EDA)
2. Text Cleaning & Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training & Evaluation (F1-Score)
5. Streamlit App Development
6. Model Deployment on AWS EC2
7. Testing & Monitoring

## 📂 Project Structure
```
flipkart-product-review-sentiment-analysis/
│
├── app/
│   └── app.py
│
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── data/
│   └── cleaned_flipkart_reviews.csv
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   └── 2_Model_Training.ipynb
│
├── requirements.txt
├── README.md
```

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/saurabhshirole1/flipkart-product-review-sentiment-analysis.git
cd flipkart-sentiment-analysis-ml
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app/app.py
```

## ☁️ AWS Deployment
* The application is deployed on **AWS EC2**
* Streamlit runs on port **8501**
* Public access via EC2 Public IP
```
http://13.235.132.119:8501/

```

## 📈 Model Evaluation
* **Evaluation Metric:** F1-Score
* The model effectively classifies customer reviews into:
   * Positive
   * Negative

## 📌 Key Learnings
* Handling real-world noisy text data
* Feature extraction for NLP tasks
* Model deployment on cloud infrastructure
* End-to-end ML project lifecycle

## 🏁 Conclusion
This project demonstrates a complete end-to-end machine learning pipeline, from data preprocessing and model training to cloud deployment. It provides valuable insights into customer sentiment and showcases practical ML deployment skills.

## 🔗 Useful Links
* **GitHub Repository:** https://github.com/saurabhshirole1/flipkart-product-review-sentiment-analysis
* **AWS Deployment Link:** http://13.235.132.119:8501/

## 🙌 Acknowledgement
I would like to thank **Innomatics Research Labs** for providing the dataset and guidance throughout the internship.