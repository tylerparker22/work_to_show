#%% REGRESSION MODEL - Predict Duration (sec) from Duration
import pandas as pd
import sqlite3
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import summary_table

# -----------------------------
# Database connection
# -----------------------------
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
#"""['IG reel' 'IG carousel' 'IG image' None]"""
POST_TYPE = "IG reel"
df = pd.read_sql_query(f"SELECT * FROM post WHERE [Post type]='{POST_TYPE}' AND FOLLOWS >=1;", conn)
conn.close()

# -----------------------------
# Convert columns to numeric
# -----------------------------
#columns in the db: ['Post ID', 'Account ID', 'Account username', 'Account name','Description', 'Duration (sec)', 'Publish time', 'Permalink','Post type', 'Data comment', 'Date', 'Views', 'Likes', 'Shares','Comments', 'Saves', 'Reach', 'Follows']
df['Views'] = pd.to_numeric(df['Views'], errors='coerce')
df['Duration (sec)'] = pd.to_numeric(df['Duration (sec)'], errors='coerce')

# Drop rows with missing Duration or Duration (sec)
df = df.dropna(subset=['Duration (sec)','Views'])

# -----------------------------
# Split into train and test sets
# -----------------------------
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train = df[df['r'] <= 0.6].copy()
test = df[df['r'] > 0.6].copy()

train.drop(columns=['r'], inplace=True)
test.drop(columns=['r'], inplace=True)

# -----------------------------
# Regression on training data
# -----------------------------
y_train = train['Views']
X_train = train['Duration (sec)']
X_train_const = sm.add_constant(X_train)  # add intercept

reg = sm.OLS(y_train, X_train_const).fit()
print(reg.summary())

# -----------------------------
# Mean Absolute Error (MAE) on test data
# -----------------------------
X_test_const = sm.add_constant(test['Views'])
y_test = test['Views']
y_pred = reg.predict(X_test_const)

mae = np.mean(np.abs(y_test - y_pred))
print("Mean Absolute Error (MAE):", mae)

# -----------------------------
# Root Mean Square Error (RMSE) on test data
# -----------------------------
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print("Root Mean Square Error (RMSE):", rmse)

# -----------------------------
# Prediction Interval Plot
# -----------------------------
x_plot = train['Duration (sec)']
st, data, ss2 = summary_table(reg, alpha=0.05)
fittedvalues = data[:, 2]
predict_ci_low, predict_ci_upp = data[:, 6:8].T

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x_plot, y_train, 'o', label='Observed')
ax.plot(x_plot, fittedvalues, '-', lw=2, label='Regression Line')
ax.plot(x_plot, predict_ci_low, 'r--', lw=2, label='Prediction Interval')
ax.plot(x_plot, predict_ci_upp, 'r--', lw=2)
ax.set_title("Predicting Duration (sec) from Views")
ax.set_xlabel("Duration (sec)")
ax.set_ylabel("Views")
ax.legend()
plt.show()
