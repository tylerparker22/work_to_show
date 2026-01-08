# -----------------------------
# Time Series: Social Media Metrics
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import ScalarFormatter

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
# Preprocess
# -----------------------------
# Split 'Publish time' column if it exists
if 'Publish time' in df.columns:
    df[['date', 'time']] = df['Publish time'].str.split(' ', expand=True)
    df['date'] = pd.to_datetime(df['date'])
else:
    df['date'] = pd.to_datetime(df['date'])

# Set 'date' as index
df = df.set_index('date')

# Select numeric columns only
numeric_cols = ['Likes', 'Views', 'Shares', 'Follows']  # adjust as needed
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaN with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# -----------------------------
# Aggregate by date (sum of numeric columns only)
# -----------------------------
daily = df[numeric_cols].groupby(df.index).sum()

# -----------------------------
# Plot: Likes with 7-day rolling average
# -----------------------------
if 'Likes' in daily.columns:
    daily['Likes_7day'] = daily['Likes'].rolling(7).mean()

    plt.figure(figsize=(12,5))
    plt.plot(daily['Likes'], marker='o', label='Likes')
    plt.plot(daily['Likes_7day'], color='red', linewidth=2, label='7-Day Avg')

    plt.title("Likes Over Time")
    plt.xlabel("Date")
    plt.ylabel("Likes")

    # Dynamic y-axis
    y_min = 0
    y_max = daily['Likes'].max() + 50
    step = max(1, int(y_max / 10))
    plt.yticks(np.arange(y_min, y_max + step, step))

    # Plain numbers (no scientific notation)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().ticklabel_format(style='plain', axis='y')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Format x-axis to include year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

# -----------------------------
# Plot all numeric metrics
# -----------------------------
available_metrics = [col for col in numeric_cols if col in daily.columns]
if available_metrics:
    ax = daily[available_metrics].plot(figsize=(12,5), marker='o')
    plt.title("Social Media Metrics Over Time")
    plt.xlabel("Date")
    plt.ylabel("Counts")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # X-axis with year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # Plain numbers for y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.show(block=True)
