import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = {
    "ticket": [
        "My internet is not working",
        "Unable to login to account",
        "Payment failed but amount deducted",
        "Need refund for wrong charge",
        "App is crashing frequently",
        "How to reset password"
    ],
    "label": [
        "Network",
        "Login",
        "Billing",
        "Billing",
        "Technical",
        "Login"
    ]
}

df = pd.DataFrame(data)

X = df["ticket"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)


import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model trained and saved successfully!")