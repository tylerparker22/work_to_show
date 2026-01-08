from dash import Dash, dcc, html, dash_table, Input, Output
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Load Data
# -----------------------------
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post", conn)
conn.close()

# -----------------------------
# Clean numeric columns
# -----------------------------
for col in ["Likes", "Views", "Shares", "Follows"]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# -----------------------------
# Prepare target
# -----------------------------
df['Follows_binary'] = (df['Follows'] > 0).astype(int)

# -----------------------------
# Auto-detect numeric features
# -----------------------------
numeric_df = df.apply(pd.to_numeric, errors='ignore')
numeric_cols = numeric_df.select_dtypes(include=['number']).columns.tolist()
remove_cols = ['Follows', 'Follows_binary', 'Post ID', 'Account ID']
features = [col for col in numeric_cols if col not in remove_cols]

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Random Forest: Predict Follower Gain", style={"textAlign": "center"}),

    html.Label("Select Post Type:"),
    dcc.RadioItems(
        id="post-type-filter",
        options=[{"label": "All", "value": "All"}] + [{"label": pt, "value": pt} for pt in df["Post type"].dropna().unique()],
        value="All",
        inline=True
    ),

    html.Hr(),
    html.H3("Feature Importance"),
    dcc.Graph(id="feature-importance"),

    html.H3("Confusion Matrix"),
    dcc.Graph(id="confusion-matrix"),

    html.H3("Example Prediction for Average Post"),
    html.Div(id="example-prediction"),

    html.H3("Data Table"),
    dash_table.DataTable(
        id="data-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "minWidth": "80px", "width": "120px", "maxWidth": "300px"}
    )
])

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("feature-importance", "figure"),
    Output("confusion-matrix", "figure"),
    Output("example-prediction", "children"),
    Output("data-table", "data"),
    Input("post-type-filter", "value")
)
def update_rf(post_type):
    filtered_df = df.copy()
    if post_type != "All":
        filtered_df = filtered_df[filtered_df["Post type"] == post_type]

    X = filtered_df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = filtered_df['Follows_binary']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=500, random_state=17)
    rf.fit(X_train, y_train)

    # Feature importance
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_fig = go.Figure()
    feat_fig.add_trace(go.Bar(
        x=np.array(features)[sorted_indices],
        y=importances[sorted_indices],
        marker_color='indianred'
    ))
    feat_fig.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Importance")

    # Confusion matrix
    predictions = rf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues',
        showscale=True
    ))
    cm_fig.update_layout(title="Confusion Matrix")

    # Example prediction for average post
    example = np.array([X.mean().values])
    predicted = rf.predict(example)[0]
    example_text = f"Prediction for an average post: {'Gain follower' if predicted==1 else 'No gain'}"

    # Update table
    table_data = filtered_df.to_dict("records")

    return feat_fig, cm_fig, example_text, table_data

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
