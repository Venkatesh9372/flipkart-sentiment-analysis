# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
import pickle

# 2. Load the Cleaned Dataset

df = pd.read_csv("data\\cleaned_flipkart_reviews.csv")

# Separate features and target variable

X = df['clean_review']  
y = df['sentiment']      

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Text Vectorization using TF-IDF

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Fit TF-IDF on training data and transform both sets
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Logistic Regression Model

# Initialize Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test_tfidf)

# Evaluate performance using F1-Score
f1_lr = f1_score(y_test, y_pred_lr)

# Print evaluation metrics
print("Logistic Regression F1-Score:", f1_lr)
print(classification_report(y_test, y_pred_lr))


# 6. Naive Bayes Model
nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate performance
f1_nb = f1_score(y_test, y_pred_nb)

# Print Naive Bayes performance
print("Naive Bayes F1-Score:", f1_nb)

# 7. Model Comparison

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression (TF-IDF)",
        "Naive Bayes (TF-IDF)"
    ],
    "F1-Score": [
        f1_lr,
        f1_nb
    ]
})

comparison_df

# 8. Save the Best Model

# Save the trained Logistic Regression model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
