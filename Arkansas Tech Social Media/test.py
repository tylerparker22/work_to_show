from dash import Dash, html, dcc, dash_table, Input, Output, State
import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import nltk

# Ensure Pandas never truncates long text in exports
pd.set_option("display.max_colwidth", None)

# -----------------------------
# Setup
# -----------------------------
nltk.download("stopwords")
DB_PATH = r"E:/Send to Katie/All Social Media/social_media.db"

# -----------------------------
# Load Data
# -----------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post", conn)
conn.close()
print(df.head())