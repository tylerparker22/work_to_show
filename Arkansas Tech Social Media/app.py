from dash import Dash, html, dcc, dash_table, Input, Output, State
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import nltk
import os
import re
import json
import time
import threading
import numpy as np
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# Load credentials
# -----------------------------
load_dotenv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Arkansas Tech Social Media/tcp_creations-ATU.env")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# -----------------------------
# Setup
# -----------------------------
pd.set_option("display.max_colwidth", None)
nltk.download("stopwords", quiet=True)

# -----------------------------
# Load Data from Supabase
# -----------------------------
response = supabase.table("Post").select("*").execute()
df = pd.DataFrame(response.data)

# -----------------------------
# Preprocess
# -----------------------------
numeric_cols = ["Likes", "Views", "Shares", "Follows"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df['Publish time'] = pd.to_datetime(df['Publish time'], errors='coerce')
df['date'] = df['Publish time'].dt.date
df = df.dropna(subset=numeric_cols + ["Description"])
df['Post type'] = df['Post type'].fillna("Unknown")
post_types = df['Post type'].unique().tolist()

# -----------------------------
# Text Features
# -----------------------------
df["Description"] = df["Description"].astype(str)
df["sentiment"]    = df["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["len_chars"]    = df["Description"].apply(len)
df["len_words"]    = df["Description"].apply(lambda x: len(x.split()))

# -----------------------------
# Clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df[numeric_cols])

# -----------------------------
# Posting Gap Columns
# -----------------------------
df = df.sort_values('Publish time')
df['days_between_posts'] = df['Publish time'].diff().dt.days.fillna(0)
df['total_posts_per_row'] = 1
df['view_like_ratio'] = df.apply(lambda row: row['Views']/row['Likes'] if row['Likes'] > 0 else 0, axis=1)

# -----------------------------
# Options for Filters
# -----------------------------
post_type_options    = [{"label": pt, "value": pt} for pt in post_types] or [{"label": "None", "value": "None"}]
cluster_options      = [{"label": str(c), "value": c} for c in sorted(df["cluster"].unique())]
days_options         = [{"label": d, "value": d} for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]
numeric_cols_options = [{"label": c, "value": c} for c in numeric_cols]

# ============================================================
# PREDICTIVE ENGINE — atu_business only
# ============================================================

def extract_description_features(text: str) -> dict:
    """Extract all predictive features from a description string."""
    text = str(text)
    tb = TextBlob(text)
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    emojis   = re.findall(r'[^\w\s,.\!\?\"\'\-]', text)
    has_question  = int('?' in text)
    has_cta       = int(any(w in text.lower() for w in ['link in bio','check out','click','swipe','follow','tag','share','comment','visit','learn more']))
    has_number    = int(bool(re.search(r'\d', text)))
    first_word_caps = int(text.split()[0].isupper()) if text.split() else 0
    return {
        "sentiment":      tb.sentiment.polarity,
        "subjectivity":   tb.sentiment.subjectivity,
        "len_chars":      len(text),
        "len_words":      len(text.split()),
        "num_hashtags":   len(hashtags),
        "num_mentions":   len(mentions),
        "num_emojis":     len(emojis),
        "has_question":   has_question,
        "has_cta":        has_cta,
        "has_number":     has_number,
        "first_word_caps":first_word_caps,
        "exclamations":   text.count('!'),
        "avg_word_len":   np.mean([len(w) for w in text.split()]) if text.split() else 0,
    }

PRED_FEATURE_COLS = [
    "sentiment","subjectivity","len_chars","len_words",
    "num_hashtags","num_mentions","num_emojis",
    "has_question","has_cta","has_number","first_word_caps",
    "exclamations","avg_word_len"
]

def build_predictor_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """Add all prediction features to a dataframe."""
    feats = source_df["Description"].apply(extract_description_features).apply(pd.Series)
    return pd.concat([source_df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

# Build atu_business subset with features
atu_df_raw = df[df["Account username"] == "atu_business"].copy()
atu_df = build_predictor_df(atu_df_raw)
atu_df = atu_df.dropna(subset=["Likes", "Views"])

def train_model(target: str):
    """Train a GradientBoosting model for a given target on atu_business data."""
    X = atu_df[PRED_FEATURE_COLS].values
    y = atu_df[target]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42))
    ])
    scores = cross_val_score(model, X, y, cv=min(5, len(atu_df)), scoring="r2")
    model.fit(X, y)
    return model, scores.mean(), scores.std()

# Train both models at startup
likes_model, likes_r2, likes_r2_std = train_model("Likes")
views_model, views_r2, views_r2_std = train_model("Views")

def predict_from_description(description: str) -> dict:
    """Given a description string, return predicted Likes and Views."""
    feats = extract_description_features(description)
    X = np.array([[feats[col] for col in PRED_FEATURE_COLS]])
    pred_likes = max(0, likes_model.predict(X)[0])
    pred_views = max(0, views_model.predict(X)[0])

    # Find similar past posts (by feature similarity)
    feat_matrix = atu_df[PRED_FEATURE_COLS].values
    new_vec = X[0]
    dists = np.linalg.norm(feat_matrix - new_vec, axis=1)
    top3_idx = np.argsort(dists)[:3]
    similar = atu_df.iloc[top3_idx][["Description", "Likes", "Views"]].to_dict("records")

    # Build tips
    tips = []
    feats_dict = feats
    avg_hashtags = atu_df["num_hashtags"].mean()
    avg_len      = atu_df["len_words"].mean()
    if feats_dict["num_hashtags"] < avg_hashtags:
        tips.append(f"📌 Add more hashtags — your posts average {avg_hashtags:.1f}, this has {feats_dict['num_hashtags']}")
    if feats_dict["has_question"] == 0:
        tips.append("❓ Adding a question tends to boost engagement — try asking your audience something")
    if feats_dict["has_cta"] == 0:
        tips.append("📣 No call-to-action detected — try adding 'link in bio', 'comment below', etc.")
    if feats_dict["len_words"] > avg_len * 1.5:
        tips.append(f"✂️ This description is long ({feats_dict['len_words']} words) — shorter ones (~{avg_len:.0f} words) tend to perform better")
    if feats_dict["sentiment"] < 0:
        tips.append("😊 Negative sentiment detected — positive descriptions tend to get more likes")
    if not tips:
        tips.append("✅ Description looks well-structured based on your historical patterns")

    return {
        "pred_likes": round(pred_likes),
        "pred_views": round(pred_views),
        "similar":    similar,
        "tips":       tips,
        "feats":      feats_dict
    }

# Percentile helpers for context
def pct_rank(value, col):
    return round((atu_df[col] < value).mean() * 100)

# ============================================================
# CLASSIFY AGENT
# ============================================================
CATEGORIES = [
    "Trending / Viral",
    "Educational / Tutorial",
    "Entertainment",
    "News / Current Events",
    "User Generated Content"
]

classify_log = []

def build_prompt(post: dict) -> str:
    return f"""
You are a social media content classifier.
Analyze the post below and respond ONLY with a valid JSON object.
No extra text. No markdown. No code blocks. Just raw JSON.

JSON format:
{{
  "category": "<one of the allowed categories>",
  "confidence": <float between 0.0 and 1.0>,
  "is_trending": <true or false>,
  "reasoning": "<one sentence explanation>"
}}

Allowed categories:
- Trending / Viral
- Educational / Tutorial
- Entertainment
- News / Current Events
- User Generated Content

Post data:
Description  : {post.get('Description', 'N/A')}
Post Type    : {post.get('Post type', 'N/A')}
Likes        : {post.get('Likes', 0)}
Views        : {post.get('Views', 0)}
Shares       : {post.get('Shares', 0)}
Follows      : {post.get('Follows', 0)}
Comments     : {post.get('Comments', 0)}
Saves        : {post.get('Saves', 0)}
Hashtags     : {post.get('hashtags', 'N/A')}
Publish Time : {post.get('Publish time', 'N/A')}
"""

def classify_post(post: dict) -> dict | None:
    try:
        response = gemini_model.generate_content(build_prompt(post))
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        if result.get("category") not in CATEGORIES:
            result["category"] = "User Generated Content"
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        return result
    except Exception as e:
        classify_log.append(f"    ERROR: {type(e).__name__}: {str(e)}")
        return None

def write_result(post_id, result: dict):
    supabase.table("Post").update({
        "category"       : result["category"],
        "confidence"     : result["confidence"],
        "is_trending"    : result["is_trending"],
        "classify_reason": result["reasoning"],
        "reasoning"      : result["reasoning"],
        "classified_at"  : datetime.utcnow().isoformat(),
        "agent_status"   : "classified"
    }).eq("Post ID", post_id).execute()

def run_classify():
    global classify_log
    classify_log = []
    classify_log.append(f"[CLASSIFY] Started — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    response = supabase.table("Post").select("*").is_("category", "null").execute()
    posts = response.data

    if not posts:
        classify_log.append("[CLASSIFY] No unclassified posts found. Done.")
        return

    classify_log.append(f"[CLASSIFY] Found {len(posts)} unclassified post(s).")
    success = 0
    failed  = 0

    for post in posts:
        post_id = post.get("Post ID")
        desc    = str(post.get("Description", ""))[:50]
        classify_log.append(f"→ Post ID={post_id} | \"{desc}...\"")

        result = classify_post(post)
        time.sleep(2)

        if result:
            write_result(post_id, result)
            classify_log.append(f"  ✓ {result['category']} | Confidence: {result['confidence']:.2f} | Trending: {result['is_trending']}")
            success += 1
        else:
            classify_log.append(f"  ✗ Failed — Post ID={post_id}")
            failed += 1

    classify_log.append(f"[CLASSIFY] Done. {success} classified, {failed} failed.")

# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Social Media Dashboard", style={"textAlign": "center"}),
    dcc.Tabs([

        # Tab 1: Average Follows
        dcc.Tab(label="Average Follows", children=[
            html.Div([
                html.Label("Select Post Type:"),
                dcc.RadioItems(id="post-type-buttons", options=post_type_options, value=post_types[0], inline=True),
                html.H2(id="average-follows-display", style={"textAlign": "center", "color": "blue"}),
                html.Div(id="corrupted-values-display", style={"marginTop": "20px"})
            ], style={"margin": "20px"})
        ]),

        # Tab 2: Regression & Clusters
        dcc.Tab(label="Regression & Clusters", children=[
            html.Div([
                html.Label("Select X-axis:"),
                dcc.Dropdown(id="x-axis-dropdown", options=numeric_cols_options, value="Likes"),
                html.Label("Select Y-axis:"),
                dcc.Dropdown(id="y-axis-dropdown", options=numeric_cols_options, value="Views"),
                html.Label("Select Graph Type:"),
                dcc.RadioItems(id="graph-type-buttons",
                               options=[{"label": "Regression", "value": "regression"},
                                        {"label": "Cluster", "value": "cluster"}],
                               value="regression", inline=True),
                html.Br(),
                html.Label("Filter by Post Type:"),
                dcc.RadioItems(id="post-type-filter",
                               options=[{"label": "All", "value": "All"}] + post_type_options,
                               value="All", inline=True),
                html.Label("Gain a Follower:"),
                dcc.RadioItems(id="gain-follower-filter",
                               options=[{"label": "All", "value": "All"}, {"label": "Yes", "value": "Gain"}],
                               value="All", inline=True),
                html.Label("Select Cluster(s):"),
                dcc.Dropdown(id="cluster-filter", options=cluster_options, multi=True, placeholder="All clusters"),
                html.Br(),
                dcc.Graph(id="dynamic-graph"),
                html.H3("Data Table"),
                dash_table.DataTable(id="dynamic-table",
                                     columns=[{"name": i, "id": i} for i in df.columns],
                                     page_action="none", filter_action="native", sort_action="native",
                                     style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "scroll"})
            ], style={"margin": "20px"})
        ]),

        # Tab 3: Random Forest
        dcc.Tab(label="Random Forest", children=[
            html.Div([
                html.Label("Select Post Type:"),
                dcc.Dropdown(id="rf-posttype-dropdown",
                             options=[{"label": "All", "value": "All"}] + post_type_options,
                             value="All"),
                html.Br(),
                html.Label("Select Target Column:"),
                dcc.Dropdown(id="rf-target-dropdown", options=numeric_cols_options, value="Likes"),
                html.Br(),
                dcc.Graph(id="rf-importance-graph")
            ], style={"margin": "20px"})
        ]),

        # Tab 4: Text Analysis
        dcc.Tab(label="Text Analysis", children=[
            html.H3("Text Features"),
            html.Div([
                html.Label("Filter by Minimum Views:"),
                dcc.Input(id="views-filter-input", type="number", placeholder="Enter min views", style={"marginRight": "20px"}),
                html.Label("Filter by Minimum Likes:"),
                dcc.Input(id="likes-filter-input", type="number", placeholder="Enter min likes")
            ], style={"marginBottom": "10px"}),
            dash_table.DataTable(id="text-analysis-table",
                                 columns=[{"name": i, "id": i} for i in ["Description","Permalink"] + numeric_cols + ["sentiment","subjectivity","len_words","len_chars"]],
                                 data=df.to_dict("records"),
                                 page_action="none", filter_action="native", sort_action="native",
                                 style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "scroll"}),
            html.Button("Download Filtered CSV", id="download-text-csv-btn", style={"marginTop": "15px"}),
            dcc.Download(id="download-text-csv")
        ]),

        # Tab 5: Time Series
        dcc.Tab(label="Time Series", children=[
            html.Div([
                html.Label("Select Metrics:"),
                dcc.Checklist(id="ts-metrics-checklist", options=numeric_cols_options, value=["Likes"], inline=True),
                html.Br(),
                html.Label("Filter by Post Type:"),
                dcc.RadioItems(id="ts-post-type-filter", options=[{"label": "All", "value": "All"}] + post_type_options,
                               value="All", inline=True),
                html.Label("Gain a Follower:"),
                dcc.RadioItems(id="ts-gain-follower-filter",
                               options=[{"label": "All", "value": "All"}, {"label": "Yes", "value": "Gain"}],
                               value="All", inline=True),
                html.Br(),
                dcc.Graph(id="time-series-graph")
            ], style={"margin": "20px"})
        ]),

        # Tab 6: Best Time to Post Heatmap
        dcc.Tab(label="Best Time to Post", children=[
            html.Div([
                html.H3("Best Time to Post by Engagement Metric"),
                html.Label("Select Metric:"),
                dcc.Dropdown(id="heatmap_metric", options=numeric_cols_options, value="Likes"),
                html.Br(),
                html.Label("Filter by Post Type:"),
                dcc.Dropdown(id="heatmap_posttype", options=[{"label": "All", "value": "All"}] + post_type_options, value="All"),
                html.Br(),
                html.Button("Select All Days", id="heatmap_select_all_days", style={"marginBottom": "10px"}),
                dcc.Checklist(id="heatmap_day_filter", options=days_options, value=[], inline=True),
                html.Br(),
                dcc.Graph(id="heatmap_graph"),
            ], style={"margin": "20px"})
        ]),

        # Tab 7: Posting Gap Analysis
        dcc.Tab(label="Posting Gap Analysis", children=[
            html.Div([
                html.H3("Days Between Posts Analysis"),
                html.Label("Filter by Post Type:"),
                dcc.Dropdown(id="posting-gap-posttype-filter", options=[{"label": "All", "value": "All"}] + post_type_options,
                             value="All"),
                html.Br(),
                html.Label("Select Metric to Visualize:"),
                dcc.RadioItems(id="posting-gap-metric",
                               options=[
                                   {"label": "Views", "value": "Views"},
                                   {"label": "Likes", "value": "Likes"},
                                   {"label": "View/Like Ratio", "value": "view_like_ratio"},
                                   {"label": "Total Posts", "value": "total_posts_per_row"}
                               ],
                               value="Views", inline=True),
                dcc.Graph(id="posting-gap-graph"),
                html.H3("Number of Posts by Days After Previous Post"),
                dcc.Graph(id="days-between-post-count-graph")
            ], style={"margin": "20px"})
        ]),

        # Tab 8: CLASSIFY Agent
        dcc.Tab(label="🤖 CLASSIFY Agent", children=[
            html.Div([
                html.H3("CLASSIFY — AI Post Categorizer", style={"textAlign": "center"}),
                html.P("Reads unclassified posts from Supabase and categorizes them using Gemini AI.",
                       style={"textAlign": "center", "color": "gray"}),
                html.Div([
                    html.Button("▶ Run CLASSIFY", id="run-classify-btn",
                                style={"fontSize": "16px", "padding": "10px 30px",
                                       "backgroundColor": "#4CAF50", "color": "white",
                                       "border": "none", "borderRadius": "8px", "cursor": "pointer"}),
                ], style={"textAlign": "center", "margin": "20px"}),
                dcc.Interval(id="classify-interval", interval=2000, n_intervals=0, disabled=True),
                html.Div(id="classify-status", style={"textAlign": "center", "color": "blue", "fontSize": "14px"}),
                html.Br(),
                html.Pre(id="classify-log",
                         style={"backgroundColor": "#1e1e1e", "color": "#00ff88",
                                "padding": "20px", "borderRadius": "8px",
                                "maxHeight": "500px", "overflowY": "scroll",
                                "fontFamily": "monospace", "fontSize": "13px",
                                "whiteSpace": "pre-wrap"})
            ], style={"margin": "20px"})
        ]),

        # Tab 9: Predictive Engine
        dcc.Tab(label="🔮 Predictive Engine", children=[
            html.Div([
                html.H3("Predictive Engine — atu_business", style={"textAlign": "center"}),
                html.P(
                    f"Trained on {len(atu_df)} atu_business posts using Gradient Boosting. "
                    f"Likes R² = {likes_r2:.2f} (±{likes_r2_std:.2f}) | "
                    f"Views R² = {views_r2:.2f} (±{views_r2_std:.2f})",
                    style={"textAlign": "center", "color": "gray", "fontSize": "13px"}
                ),

                # Input area
                html.Div([
                    html.Label("Write your post description:", style={"fontWeight": "bold"}),
                    dcc.Textarea(
                        id="pred-description-input",
                        placeholder="Type your post caption here...",
                        style={"width": "100%", "height": "120px", "fontSize": "14px",
                               "borderRadius": "8px", "padding": "10px", "marginTop": "8px"}
                    ),
                    html.Br(),
                    html.Button(
                        "🔮 Predict Engagement",
                        id="pred-run-btn",
                        style={"fontSize": "15px", "padding": "10px 28px",
                               "backgroundColor": "#6c3fc5", "color": "white",
                               "border": "none", "borderRadius": "8px",
                               "cursor": "pointer", "marginTop": "10px"}
                    ),
                ], style={"marginBottom": "30px"}),

                # Results
                html.Div(id="pred-results-div"),

            ], style={"margin": "30px", "maxWidth": "900px"})
        ]),

    ])
])

# ============================================================
# CALLBACKS
# ============================================================

# Average Follows
@app.callback(
    Output("average-follows-display", "children"),
    Output("corrupted-values-display", "children"),
    Input("post-type-buttons", "value")
)
def update_average(post_type):
    filtered_df = df[df["Post type"] == post_type]
    avg_follows = filtered_df["Follows"].mean()
    avg_text = f"Average Follows for {post_type}: {avg_follows:.2f}"
    bad_rows = filtered_df[filtered_df["Follows"].isna()]
    corrupted_text = html.Div([
        html.H4("Corrupted Follows values found:"),
        dash_table.DataTable(data=bad_rows.to_dict("records"), page_action="none",
                             style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "scroll"})
    ]) if not bad_rows.empty else html.Div()
    return avg_text, corrupted_text

# Regression & Clusters
@app.callback(
    Output("dynamic-graph", "figure"),
    Output("dynamic-table", "data"),
    Input("graph-type-buttons", "value"),
    Input("x-axis-dropdown", "value"),
    Input("y-axis-dropdown", "value"),
    Input("cluster-filter", "value"),
    Input("post-type-filter", "value"),
    Input("gain-follower-filter", "value")
)
def update_dynamic_graph(graph_type, x_col, y_col, selected_clusters, post_type, gain_filter):
    filtered_df = df.copy()
    if selected_clusters:
        filtered_df = filtered_df[filtered_df["cluster"].isin(selected_clusters)]
    if post_type != "All":
        filtered_df = filtered_df[filtered_df["Post type"] == post_type]
    if gain_filter == "Gain":
        filtered_df = filtered_df[filtered_df["Follows"] >= 1]
    if graph_type == "regression":
        pred_frame = sm.OLS(filtered_df[y_col], sm.add_constant(filtered_df[x_col])).fit().get_prediction(sm.add_constant(filtered_df[x_col])).summary_frame(alpha=0.05)
        fig = px.scatter(filtered_df, x=x_col, y=y_col, hover_data=["Description","Permalink"], title=f"Regression: {y_col} ~ {x_col}")
        fig.add_scatter(x=filtered_df[x_col], y=pred_frame["mean"], mode="lines", name="OLS Fit", line=dict(color="blue"))
        fig.add_scatter(x=filtered_df[x_col], y=pred_frame["obs_ci_lower"], mode="lines", name="95% PI Lower", line=dict(color="red", dash="dash"))
        fig.add_scatter(x=filtered_df[x_col], y=pred_frame["obs_ci_upper"], mode="lines", name="95% PI Upper", line=dict(color="red", dash="dash"))
    else:
        fig = px.scatter(filtered_df, x=x_col, y=y_col, color="cluster",
                         hover_data=["Description","Permalink","sentiment","subjectivity","len_words","len_chars"],
                         title=f"Cluster: {y_col} vs {x_col}")
    return fig, filtered_df.to_dict("records")

# Random Forest
@app.callback(
    Output("rf-importance-graph", "figure"),
    Input("rf-target-dropdown", "value"),
    Input("rf-posttype-dropdown", "value")
)
def update_rf_importance(target_col, post_type):
    filtered_df = df.copy()
    if post_type != "All":
        filtered_df = filtered_df[filtered_df["Post type"] == post_type]
    features = [c for c in numeric_cols if c != target_col]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(filtered_df[features], filtered_df[target_col])
    importance_df = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    return px.bar(importance_df, x="Feature", y="Importance", title=f"Random Forest Feature Importance for {target_col}")

# Time Series
@app.callback(
    Output("time-series-graph", "figure"),
    Input("ts-metrics-checklist", "value"),
    Input("ts-post-type-filter", "value"),
    Input("ts-gain-follower-filter", "value")
)
def update_time_series(metrics, post_type, gain_filter):
    filtered_df = df.copy()
    if post_type != "All":
        filtered_df = filtered_df[filtered_df["Post type"] == post_type]
    if gain_filter == "Gain":
        filtered_df = filtered_df[filtered_df["Follows"] >= 1]
    filtered_df = filtered_df.sort_values("Publish time")
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(x=filtered_df["Publish time"], y=filtered_df[metric], mode="lines+markers", name=metric))
        fig.add_trace(go.Scatter(x=filtered_df["Publish time"], y=filtered_df[metric].rolling(7).mean(), mode="lines", name=f"{metric} 7-day Avg", line=dict(dash="dash")))
    fig.update_layout(title="Time Series of Social Media Metrics", xaxis_title="Date", yaxis_title="Count", xaxis=dict(tickformat="%b %d, %Y"))
    return fig

# Text Analysis
@app.callback(
    Output("text-analysis-table", "data"),
    Input("views-filter-input", "value"),
    Input("likes-filter-input", "value")
)
def filter_text_analysis(min_views, min_likes):
    filtered_df = df.copy()
    if min_views is not None:
        filtered_df = filtered_df[filtered_df["Views"] >= min_views]
    if min_likes is not None:
        filtered_df = filtered_df[filtered_df["Likes"] >= min_likes]
    return filtered_df.to_dict("records")

@app.callback(
    Output("download-text-csv", "data"),
    Input("download-text-csv-btn", "n_clicks"),
    State("views-filter-input", "value"),
    State("likes-filter-input", "value"),
    prevent_initial_call=True
)
def download_text_file(n_clicks, min_views, min_likes):
    filtered_df = df.copy()
    if min_views is not None:
        filtered_df = filtered_df[filtered_df["Views"] >= min_views]
    if min_likes is not None:
        filtered_df = filtered_df[filtered_df["Likes"] >= min_likes]
    cols = ["Description"] + [c for c in filtered_df.columns if c != "Description"]
    return dcc.send_data_frame(filtered_df[cols].to_csv, "text_analysis_filtered.csv", index=False)

# Heatmap Select All Days
@app.callback(
    Output("heatmap_day_filter", "value"),
    Input("heatmap_select_all_days", "n_clicks"),
    prevent_initial_call=True
)
def select_all_days(n):
    return [d['value'] for d in days_options]

# Heatmap
@app.callback(
    Output("heatmap_graph", "figure"),
    Input("heatmap_metric", "value"),
    Input("heatmap_posttype", "value"),
    Input("heatmap_day_filter", "value")
)
def update_heatmap(metric, post_type, selected_days):
    dff = df.copy()
    if post_type != "All":
        dff = dff[dff["Post type"] == post_type]
    dff["post_time"] = dff["Publish time"].dt.hour
    dff["post_day"]  = dff["Publish time"].dt.day_name()
    if selected_days:
        dff = dff[dff["post_day"].isin(selected_days)]
    pivot = dff.groupby(["post_day", "post_time"])[metric].mean().reset_index()
    fig = px.density_heatmap(pivot, x="post_time", y="post_day", z=metric,
                             color_continuous_scale=["blue","green","yellow","red"],
                             nbinsx=24, title=f"Best Time to Post for {metric}")
    fig.update_layout(xaxis_title="Hour of Day (0-23)", yaxis_title="Day of Week")
    return fig

# Posting Gap
@app.callback(
    Output("posting-gap-graph", "figure"),
    Input("posting-gap-metric", "value"),
    Input("posting-gap-posttype-filter", "value")
)
def update_posting_gap_graph(metric, post_type):
    dff = df.copy()
    if post_type != "All":
        dff = dff[dff["Post type"] == post_type]
    dff = dff.sort_values("Publish time")
    dff['days_between_posts'] = dff['Publish time'].diff().dt.days.fillna(0)
    summary = dff.groupby('days_between_posts', as_index=False).agg({
        'total_posts_per_row': 'sum', 'Views': 'mean', 'Likes': 'mean', 'view_like_ratio': 'mean'
    })
    return px.line(summary, x='days_between_posts', y=metric, markers=True,
                   title=f"{metric} vs Days After Previous Post")

@app.callback(
    Output("days-between-post-count-graph", "figure"),
    Input("posting-gap-posttype-filter", "value")
)
def update_days_between_post_count(post_type):
    dff = df.copy()
    if post_type != "All":
        dff = dff[dff["Post type"] == post_type]
    dff = dff.sort_values("Publish time")
    dff['days_between_posts'] = dff['Publish time'].diff().dt.days.fillna(0)
    counts = dff['days_between_posts'].value_counts().reset_index()
    counts.columns = ['Days After Previous Post', 'Number of Posts']
    counts = counts.sort_values('Days After Previous Post')
    return px.bar(counts, x='Days After Previous Post', y='Number of Posts',
                  text='Number of Posts', title="Number of Posts by Days After Previous Post")

# CLASSIFY Agent Callbacks
@app.callback(
    Output("classify-interval", "disabled"),
    Output("classify-status", "children"),
    Input("run-classify-btn", "n_clicks"),
    prevent_initial_call=True
)
def start_classify(n_clicks):
    thread = threading.Thread(target=run_classify, daemon=True)
    thread.start()
    return False, "⏳ CLASSIFY is running..."

@app.callback(
    Output("classify-log", "children"),
    Output("classify-interval", "disabled", allow_duplicate=True),
    Output("classify-status", "children", allow_duplicate=True),
    Input("classify-interval", "n_intervals"),
    prevent_initial_call=True
)
def update_classify_log(n):
    log_text = "\n".join(classify_log)
    done = any("Done." in line for line in classify_log)
    if done:
        return log_text, True, "✅ CLASSIFY complete!"
    return log_text, False, "⏳ CLASSIFY is running..."

# ============================================================
# PREDICTIVE ENGINE CALLBACK
# ============================================================
@app.callback(
    Output("pred-results-div", "children"),
    Input("pred-run-btn", "n_clicks"),
    State("pred-description-input", "value"),
    prevent_initial_call=True
)
def run_prediction(n_clicks, description):
    if not description or not description.strip():
        return html.P("⚠️ Please enter a description first.", style={"color": "orange"})

    result = predict_from_description(description)
    likes_pct = pct_rank(result["pred_likes"], "Likes")
    views_pct = pct_rank(result["pred_views"], "Views")

    # Gauge-style color for predictions
    def pct_color(p):
        if p >= 75: return "#2ecc71"
        if p >= 40: return "#f39c12"
        return "#e74c3c"

    # Feature breakdown chart
    feats = result["feats"]
    feat_labels = {
        "sentiment": "Sentiment", "subjectivity": "Subjectivity",
        "len_words": "Word Count", "num_hashtags": "Hashtags",
        "num_emojis": "Emojis", "has_question": "Has Question",
        "has_cta": "Has CTA", "num_mentions": "Mentions"
    }
    feat_fig = go.Figure(go.Bar(
        x=[feat_labels.get(k, k) for k in feat_labels],
        y=[feats.get(k, 0) for k in feat_labels],
        marker_color=["#6c3fc5" if feats.get(k, 0) > 0 else "#cccccc" for k in feat_labels],
        text=[round(feats.get(k, 0), 2) for k in feat_labels],
        textposition="outside"
    ))
    feat_fig.update_layout(
        title="Description Feature Breakdown",
        height=300,
        margin=dict(t=40, b=20),
        yaxis_title="Value"
    )

    # Similar posts table
    similar_rows = [
        html.Tr([
            html.Td(r["Description"][:80] + "...", style={"fontSize": "12px", "padding": "6px"}),
            html.Td(str(r["Likes"]), style={"textAlign": "center", "padding": "6px"}),
            html.Td(str(r["Views"]), style={"textAlign": "center", "padding": "6px"}),
        ])
        for r in result["similar"]
    ]

    return html.Div([

        # Prediction cards
        html.Div([
            html.Div([
                html.H2(f"{result['pred_likes']:,}", style={"color": pct_color(likes_pct), "margin": "0", "fontSize": "42px"}),
                html.P("Predicted Likes", style={"margin": "4px 0", "fontWeight": "bold"}),
                html.P(f"Better than {likes_pct}% of your posts", style={"color": "gray", "fontSize": "12px", "margin": "0"})
            ], style={"textAlign": "center", "padding": "20px 40px", "backgroundColor": "#f9f9f9",
                      "borderRadius": "12px", "border": "2px solid #e0e0e0", "flex": "1"}),

            html.Div([
                html.H2(f"{result['pred_views']:,}", style={"color": pct_color(views_pct), "margin": "0", "fontSize": "42px"}),
                html.P("Predicted Views", style={"margin": "4px 0", "fontWeight": "bold"}),
                html.P(f"Better than {views_pct}% of your posts", style={"color": "gray", "fontSize": "12px", "margin": "0"})
            ], style={"textAlign": "center", "padding": "20px 40px", "backgroundColor": "#f9f9f9",
                      "borderRadius": "12px", "border": "2px solid #e0e0e0", "flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "24px"}),

        # Tips
        html.Div([
            html.H4("💡 Optimization Tips", style={"marginBottom": "10px"}),
            html.Ul([html.Li(tip, style={"marginBottom": "6px", "fontSize": "14px"}) for tip in result["tips"]])
        ], style={"backgroundColor": "#fffbe6", "border": "1px solid #ffe58f",
                  "borderRadius": "10px", "padding": "16px", "marginBottom": "24px"}),

        # Feature breakdown chart
        dcc.Graph(figure=feat_fig, config={"displayModeBar": False}),

        # Similar past posts
        html.Div([
            html.H4("📋 Most Similar Past Posts", style={"marginBottom": "10px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Description", style={"textAlign": "left", "padding": "6px"}),
                    html.Th("Likes", style={"padding": "6px"}),
                    html.Th("Views", style={"padding": "6px"})
                ])),
                html.Tbody(similar_rows)
            ], style={"width": "100%", "borderCollapse": "collapse",
                      "border": "1px solid #e0e0e0", "fontSize": "13px"})
        ], style={"marginTop": "20px"})

    ])

# ============================================================
# Run App
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8052)
