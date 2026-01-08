import sqlite3
import pandas as pd
import glob
import os

DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"

# -----------------------------
# Define columns
# -----------------------------
story_columns = [
    "Post ID",
    "Account ID",
    "Account username",
    "Account name",
    "Description",
    "Duration (sec)",
    "Publish time",
    "Permalink",
    "Post type",
    "Data comment",
    "Date",
    "Views",
    "Reach",
    "Likes",
    "Shares",
    "Profile visits",
    "Replies",
    "Sticker taps",
    "Navigation",
    "Link clicks"
]

post_columns = [
    "Post ID",
    "Account ID",
    "Account username",
    "Account name",
    "Description",
    "Duration (sec)",
    "Publish time",
    "Permalink",
    "Post type",
    "Data comment",
    "Date",
    "Views",
    "Likes",
    "Shares",
    "Comments",
    "Saves",
    "Reach",
    "Follows"
]

# -----------------------------
# Convert columns to SQL schema
# -----------------------------
def make_sql_schema(cols):
    return ",\n".join([f'"{c}" TEXT' for c in cols])

#create 'story' table
story_schema = make_sql_schema(story_columns)
#create 'post' table
post_schema = make_sql_schema(post_columns)

# -----------------------------
# Create database + tables
# -----------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS story;")
cur.execute("DROP TABLE IF EXISTS post;")

cur.execute(f"CREATE TABLE story ({story_schema});")
cur.execute(f"CREATE TABLE post ({post_schema});")

conn.commit()
print("Story and Post tables created successfully!")

# -----------------------------
# Paths to CSV folders
# -----------------------------
story_folder = "E:/Send to Katie/All Social Media/ALL SOCIAL DATA/Story"
post_folder = "E:/Send to Katie/All Social Media/ALL SOCIAL DATA/Post"

# -----------------------------
# Helper function to clean/reorder columns
# -----------------------------
def prepare_df(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    return df

# -----------------------------
# Load Story CSVs
# -----------------------------
story_files = glob.glob(os.path.join(story_folder, "*.csv"))
for file in story_files:
    df = pd.read_csv(file)
    df = prepare_df(df, story_columns)
    df.to_sql("story", conn, if_exists="append", index=False)
    print(f"Loaded {os.path.basename(file)} into story table")

# -----------------------------
# Load Post CSVs
# -----------------------------
post_files = glob.glob(os.path.join(post_folder, "*.csv"))
for file in post_files:
    df = pd.read_csv(file)
    df = prepare_df(df, post_columns)
    df.to_sql("post", conn, if_exists="append", index=False)
    print(f"Loaded {os.path.basename(file)} into post table")

conn.close()
print("All files loaded successfully!")
