#%% Average Follows for a Post type
import sqlite3
import pandas as pd

# -----------------------------
# Database connection
# -----------------------------
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)

# -----------------------------
# Read data for a specific Post type
# -----------------------------
#"""['IG reel' 'IG carousel' 'IG image' None]"""
POST_TYPE = "IG image"
df = pd.read_sql_query(
    f"SELECT * FROM post WHERE [Post type]='{POST_TYPE}' AND FOLLOWS >=1;",
    conn
)
conn.close()

# -----------------------------
# Convert 'Follows' to numeric
# -----------------------------
df['Follows'] = pd.to_numeric(df['Follows'], errors='coerce').fillna(0)
df['Follows_binary'] = df['Follows'].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# FIX: Convert Follows to numeric (bad strings → NaN)
# -----------------------------
df["Follows"] = pd.to_numeric(df["Follows"], errors="coerce")

# -----------------------------
# Compute average Follows
# -----------------------------
average_Follows = df["Follows"].mean()

print(f"Average Follows for {POST_TYPE}: {average_Follows}")

# -----------------------------
# OPTIONAL: Show corrupted Follows values
# -----------------------------
bad_rows = df[df["Follows"].isna()]
if not bad_rows.empty:
    print("\nCorrupted Follows values found:")
    print(bad_rows[["Follows"]])
