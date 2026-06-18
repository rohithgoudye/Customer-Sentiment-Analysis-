import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(
    r"C:\Users\ROHIT\OneDrive\Desktop\Customer-Sentiment-Analysis\1429_1.csv\1429_1.csv",
    low_memory=False
)

# Keep required columns
df = df[['reviews.text', 'reviews.rating']]

# Remove missing values
df.dropna(inplace=True)

# Convert ratings into sentiment labels
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["reviews.rating"].apply(get_sentiment)

# Check sentiment distribution
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# Features and Labels
X = df["reviews.text"]
y = df["sentiment"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save Vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel Saved Successfully!")