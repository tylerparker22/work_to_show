from dash import Dash, html, dcc, Output, Input
import sqlite3
import pandas as pd

# -----------------------------
# Database path
# -----------------------------
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"

# -----------------------------
# Load data
# -----------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post WHERE FOLLOWS >= 1", conn)
conn.close()

# Ensure numeric
df['Follows'] = pd.to_numeric(df['Follows'], errors='coerce').fillna(0)
df['Post type'] = df['Post type'].fillna("Unknown")

# Get unique post types for buttons
post_types = df['Post type'].unique().tolist()

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Average Follows per Post Type", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Post Type:"),
        dcc.RadioItems(
            id="post-type-buttons",
            options=[{"label": pt, "value": pt} for pt in post_types],
            value=post_types[0],  # default selection
            inline=True
        )
    ], style={"marginBottom": "20px"}),

    html.H2(id="average-follows-display", style={"textAlign": "center", "color": "blue"}),

    html.Div(id="corrupted-values-display", style={"marginTop": "20px"})
])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output("average-follows-display", "children"),
    Output("corrupted-values-display", "children"),
    Input("post-type-buttons", "value")
)
def update_average(post_type):
    filtered_df = df[df["Post type"] == post_type].copy()
    
    avg_follows = filtered_df["Follows"].mean()
    avg_text = f"Average Follows for {post_type}: {avg_follows:.2f}"

    # Show any corrupted values
    bad_rows = filtered_df[filtered_df["Follows"].isna()]
    if not bad_rows.empty:
        corrupted_text = html.Div([
            html.H4("Corrupted Follows values found:"),
            dcc.Markdown(bad_rows[["Follows"]].to_markdown())
        ])
    else:
        corrupted_text = html.Div()

    return avg_text, corrupted_text

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
