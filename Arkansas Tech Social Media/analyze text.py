from dash import Dash, dcc, html, dash_table, Input, Output
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
import statsmodels.api as sm
import plotly.express as px

# -----------------------------
# Setup
# -----------------------------
nltk.download("stopwords")

DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"

# -----------------------------
# Load Data
# -----------------------------
conn = sqlite3.connect(DB_PATH)
query = "SELECT * FROM post"
df = pd.read_sql_query(query, conn)
conn.close()

# -----------------------------
# Clean numeric columns
# -----------------------------
for col in ["Likes", "Views", "Shares", "Follows"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Likes", "Views", "Shares", "Follows", "Description"])

# -----------------------------
# Text Features
# -----------------------------
df["Description"] = df["Description"].astype(str)
df["sentiment"] = df["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["len_chars"] = df["Description"].apply(len)
df["len_words"] = df["Description"].apply(lambda x: len(x.split()))

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=300)
X_tfidf = tfidf.fit_transform(df["Description"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)

# -----------------------------
# Regression: Predict Likes from Text
# -----------------------------
X = tfidf_df
y = df["Likes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sk_model = LinearRegression().fit(X_train, y_train)
print("R² (text → likes):", sk_model.score(X_test, y_test))

# -----------------------------
# StatsModels OLS: Predict Views from Likes
# -----------------------------
X2 = sm.add_constant(df["Likes"])
y2 = df["Views"]
ols_model = sm.OLS(y2, X2).fit()
pred_frame = ols_model.get_prediction(X2).summary_frame(alpha=0.05)

# Regression table
regression_table_df = df[["Likes", "Views", "Follows", "Permalink", "Description"]].copy()
regression_table_df["OLS_Fit"] = pred_frame["mean"]
regression_table_df["PI_Lower"] = pred_frame["obs_ci_lower"]
regression_table_df["PI_Upper"] = pred_frame["obs_ci_upper"]

# -----------------------------
# Clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X_tfidf)

cluster_table_df = df[[
    "Likes", "Views", "Shares", "Follows", "cluster", "Permalink",
    "Description", "sentiment", "subjectivity", "len_words", "len_chars"
]].copy()

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__, suppress_callback_exceptions=True)

# Regression plot
regression_fig = px.scatter(
    df,
    x="Likes",
    y="Views",
    title="OLS Regression: Likes → Views",
    labels={"Likes": "Likes", "Views": "Views"},
    hover_data=["Description", "Permalink"]
)
regression_fig.add_scatter(
    x=df["Likes"], y=pred_frame["mean"],
    mode="lines", name="OLS Fit", line=dict(color="blue")
)
regression_fig.add_scatter(
    x=df["Likes"], y=pred_frame["obs_ci_lower"],
    mode="lines", name="95% PI Lower", line=dict(color="red", dash="dash")
)
regression_fig.add_scatter(
    x=df["Likes"], y=pred_frame["obs_ci_upper"],
    mode="lines", name="95% PI Upper", line=dict(color="red", dash="dash")
)

# Cluster plot
cluster_fig = px.scatter(
    df,
    x="Views",
    y="Likes",
    color="cluster",
    color_discrete_sequence=['red', 'blue', 'green', 'orange'],
    hover_data=["Description", "Permalink", "sentiment", "subjectivity", "len_words", "len_chars"],
    title="Cluster Visualization: Views vs Likes"
)

# -----------------------------
# App Layout
# -----------------------------
app.layout = html.Div([
    html.H1("Social Media Post Analysis", style={"textAlign": "center"}),

    html.Div([
        html.Label("Filter by Post Type:"),
        dcc.RadioItems(
            id="post-type-filter",
            options=[{"label": "All", "value": "All"}] + [{"label": pt, "value": pt} for pt in df["Post type"].unique()],
            value="All",
            inline=True
        ),
        html.Label("Gain a Follower:"),
        dcc.RadioItems(
            id="gain-follower-filter",
            options=[{"label": "All", "value": "All"}, {"label": "Yes", "value": "Gain"}],
            value="All",
            inline=True
        )
    ], style={"margin": "20px 0"}),

    dcc.Tabs([
        dcc.Tab(label="Regression", children=[
            dcc.Graph(id="regression-graph", figure=regression_fig),
            html.H3("Regression Data"),
            dash_table.DataTable(
                id="regression-table",
                columns=[{"name": i, "id": i, "presentation": "markdown"} for i in regression_table_df.columns],
                data=regression_table_df.to_dict("records"),
                page_size=15,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "80px", "width": "120px", "maxWidth": "300px"}
            )
        ]),
        dcc.Tab(label="Clusters", children=[
            html.Label("Select Cluster(s):"),
            dcc.Dropdown(
                id="cluster-filter",
                options=[{"label": str(c), "value": c} for c in sorted(df["cluster"].unique())],
                value=None,
                placeholder="All clusters",
                multi=True
            ),
            dcc.Graph(id="cluster-graph", figure=cluster_fig),
            html.H3("Cluster Data"),
            dash_table.DataTable(
                id="cluster-table",
                columns=[{"name": i, "id": i, "presentation": "markdown"} for i in cluster_table_df.columns],
                data=cluster_table_df.to_dict("records"),
                page_size=15,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "minWidth": "80px", "width": "120px", "maxWidth": "300px"}
            )
        ])
    ])
])

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("cluster-graph", "figure"),
    Output("cluster-table", "data"),
    Output("regression-graph", "figure"),
    Output("regression-table", "data"),
    Input("cluster-filter", "value"),
    Input("post-type-filter", "value"),
    Input("gain-follower-filter", "value")
)
def update_dashboard(selected_clusters, post_type, gain_filter):
    filtered_df = df.copy()

    # Filter by cluster
    if selected_clusters:
        filtered_df = filtered_df[filtered_df["cluster"].isin(selected_clusters)]

    # Filter by post type
    if post_type != "All":
        filtered_df = filtered_df[filtered_df["Post type"] == post_type]

    # Filter by gain a follower
    if gain_filter == "Gain":
        filtered_df = filtered_df[filtered_df["Follows"] >= 1]

    # Update cluster figure
    cluster_fig = px.scatter(
        filtered_df,
        x="Views",
        y="Likes",
        color="cluster",
        color_discrete_sequence=['red', 'blue', 'green', 'orange'],
        hover_data=["Description", "Permalink", "sentiment", "subjectivity", "len_words", "len_chars"],
        title="Cluster Visualization: Views vs Likes"
    )

    # Update cluster table
    cluster_table_data = filtered_df[[
        "Likes", "Views", "Shares", "Follows", "cluster", "Permalink",
        "Description", "sentiment", "subjectivity", "len_words", "len_chars"
    ]].to_dict("records")

    # Update regression figure
    regression_fig = px.scatter(
        filtered_df,
        x="Likes",
        y="Views",
        hover_data=["Description", "Permalink"],
        title="OLS Regression: Likes → Views"
    )
    pred_frame_filtered = sm.OLS(filtered_df["Views"], sm.add_constant(filtered_df["Likes"])).fit().get_prediction(sm.add_constant(filtered_df["Likes"])).summary_frame(alpha=0.05)
    regression_fig.add_scatter(x=filtered_df["Likes"], y=pred_frame_filtered["mean"], mode="lines", name="OLS Fit", line=dict(color="blue"))
    regression_fig.add_scatter(x=filtered_df["Likes"], y=pred_frame_filtered["obs_ci_lower"], mode="lines", name="95% PI Lower", line=dict(color="red", dash="dash"))
    regression_fig.add_scatter(x=filtered_df["Likes"], y=pred_frame_filtered["obs_ci_upper"], mode="lines", name="95% PI Upper", line=dict(color="red", dash="dash"))

    # Update regression table
    regression_table_data = filtered_df[["Likes", "Views", "Follows", "Permalink", "Description"]].copy()
    regression_table_data["OLS_Fit"] = pred_frame_filtered["mean"]
    regression_table_data["PI_Lower"] = pred_frame_filtered["obs_ci_lower"]
    regression_table_data["PI_Upper"] = pred_frame_filtered["obs_ci_upper"]
    regression_table_data = regression_table_data.to_dict("records")

    return cluster_fig, cluster_table_data, regression_fig, regression_table_data

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
