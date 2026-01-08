from dash import Dash, html, dash_table, dcc, Input, Output
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# ---------------------------------------
# Load data from SQLite
# ---------------------------------------
DB_PATH = r"E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)

# Load ONLY reels
POST_TYPE = "IG reel"
df = pd.read_sql_query(f"SELECT * FROM post WHERE [Post type]='{POST_TYPE}'", conn)
conn.close()

# ---------------------------------------
# Clean numeric columns for clustering
# ---------------------------------------
numeric_cols = ["Views", "Reach", "Likes", "Shares", "Profile visits",
                "Replies", "Sticker taps", "Navigation", "Link clicks"]

# Keep only columns that exist
numeric_cols = [col for col in numeric_cols if col in df.columns]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------------------------------
# KMeans Clustering
# ---------------------------------------
K = 4  # number of clusters (change if needed)
kmeans = KMeans(n_clusters=K, random_state=42)
df["cluster"] = kmeans.fit_predict(df[numeric_cols])

# Ensure Post ID is string
df['Post ID'] = df['Post ID'].astype(str)

# ---------------------------------------
# Build Dash App
# ---------------------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("IG Reel Clusters Viewer"),

    html.Label("Select Cluster:"),
    dcc.Dropdown(
        id="cluster_filter",
        options=[{"label": f"Cluster {i}", "value": i} for i in sorted(df["cluster"].unique())],
        value=None,   # show all by default
        clearable=True
    ),

    dash_table.DataTable(
        id="table",
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict("records"),
        page_size=20,
        filter_action="native",   # lets you filter Post ID
        sort_action="native",
        style_table={"overflowX": "auto"},
    )
])


# ---------------------------------------
# Callback to filter by cluster
# ---------------------------------------
@app.callback(
    Output("table", "data"),
    Input("cluster_filter", "value")
)
def update_table(selected_cluster):

    if selected_cluster is None:
        filtered_df = df
    else:
        filtered_df = df[df["cluster"] == selected_cluster]

    return filtered_df.to_dict("records")


# ---------------------------------------
# Run App
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
