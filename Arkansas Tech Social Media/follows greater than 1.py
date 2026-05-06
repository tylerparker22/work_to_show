from dash import Dash, html, dash_table
import sqlite3
import pandas as pd

# -----------------------------
# Connect to DB and filter posts with at least 1 follower
# -----------------------------
DB_PATH = r"E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)

# Select only relevant columns for clarity
query = """
SELECT [Post ID], [Account username], [Account name], [Date], [Follows]
FROM post
WHERE Follows >= 1;
"""
df = pd.read_sql_query(query, conn)
conn.close()

# -----------------------------
# Create Dash app
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Posts That Gained Followers"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=20,
        filter_action="native",  # lets you filter in the web browser
        sort_action="native",    # lets you sort columns
        style_table={"overflowX": "auto"}
    )
])

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
