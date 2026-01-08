# -----------------------------
# Sentiment and Subjectivity Analysis and Regression
# -----------------------------

import pandas as pd
import sqlite3
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Setup
# -----------------------------
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"

# -----------------------------
# Load Data
# -----------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post", conn)
conn.close()

# -----------------------------
# Text Features
# -----------------------------
df["Description"] = df["Description"].astype(str)
df["sentiment"] = df["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["len_chars"] = df["Description"].apply(len)
df["len_words"] = df["Description"].apply(lambda x: len(x.split()))

# -----------------------------
# Example regression: Predict Views using sentiment and subjectivity
# -----------------------------
# Drop rows with missing Views
df = df[pd.to_numeric(df['Views'], errors='coerce').notna()]
df['Views'] = df['Views'].astype(float)

# Define predictors and target
X = df[['sentiment', 'subjectivity']]  # predictors
y = df['Views']                        # target

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X, y)

# Regression coefficients
print("Intercept:", lr.intercept_)
print("Coefficients:", dict(zip(X.columns, lr.coef_)))

# Predicted Views
df['Views_pred'] = lr.predict(X)

# -----------------------------
# Optional: Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Views'], df['Views_pred'], alpha=0.6)
plt.plot([df['Views'].min(), df['Views'].max()],
         [df['Views'].min(), df['Views'].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Regression: Views ~ Sentiment + Subjectivity")
plt.show()
