#%% All Social Media Post Cluster
import sqlite3
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Database connection
# -----------------------------
DB_PATH = "D:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)

# -----------------------------
# Read data for a specific Post type
# Replace 'IG reel' with any other post type if needed
#"""['IG reel' 'IG carousel' 'IG image' None]"""
# -----------------------------
POST_TYPE = ['IG reel', 'IG carousel', 'IG image']

# Convert list to a string like: "'IG reel','IG carousel','IG image'"
types_str = ','.join(f"'{t}'" for t in POST_TYPE)

# Use IN clause in SQL
query = f"SELECT * FROM post WHERE [Post type] IN ({types_str})"
df = pd.read_sql_query(query, conn)


conn.close()

# Count and sqrt (for later use in range or clustering)
count = df.shape[0]
count_sqrt = math.sqrt(count)
print(f"{POST_TYPE} count: {count}, sqrt: {count_sqrt}")

# -----------------------------
# Check unique Post types in the database
# -----------------------------
# (optional if you want all types)
# df_all = pd.read_sql_query("SELECT * FROM post", conn)
# unique_post_types = df_all['Post type'].unique()
# print("Unique Post types:", unique_post_types)

# -----------------------------
# Convert relevant columns to numeric
# -----------------------------
numeric_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df[numeric_cols] = df[numeric_cols].fillna(0)

# -----------------------------
# K-Means clustering
# -----------------------------
tocluster = df[["Views", "Likes"]]

# Test CH index for clusters from 2 to 6
for i in range(2, 11): #change y axis to the sqrt of count
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(tocluster)
    ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
    print(f"Calinski–Harabasz Index for {i} clusters: {ch_score}")

# -----------------------------
# Choose 5 clusters (or any based on CH)
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(tocluster)
clusters = kmeans.predict(tocluster)

# Add cluster membership to DataFrame
df['cluster'] = clusters

# -----------------------------
# Cluster summary
# -----------------------------
cluster_means = df.pivot_table(index="cluster", values=["Views", "Likes"], aggfunc="mean")
cluster_counts = df.pivot_table(index="cluster", values=["Shares"], aggfunc="count")

print("Cluster Means:\n", cluster_means)
print("\nNumber of Shares per Cluster:\n", cluster_counts)

# -----------------------------
# Plot clusters
# -----------------------------
df["Clusters"] = clusters
plt.rc('font', size=8)
sns.set_style('white')
customPalette = ['red', 'blue', 'green', 'yellow', 'black']
sns.set_palette(customPalette)

sns.lmplot(
    data=df, 
    x='Views', 
    y='Likes', 
    hue='Clusters', 
    fit_reg=False, 
    legend=True
)

plt.show()

# -----------------------------
# View Post IDs from cluster 3 outliers
# -----------------------------
CLUSTER_NUM = 1  # change this to the cluster you want

# Filter rows in that cluster
cluster_rows = df[df['cluster'] == CLUSTER_NUM]

# Optional: define outliers using percentile (e.g., top 95% of Views or Likes)
likes_thresh = cluster_rows['Likes'].quantile(0.95)
views_thresh = cluster_rows['Views'].quantile(0.95)
outliers = cluster_rows[
    (cluster_rows['Likes'] >= likes_thresh) | (cluster_rows['Views'] >= views_thresh)
]

# Show only Post IDs
print(f"Post IDs in cluster {CLUSTER_NUM} outliers:")
print(outliers['Post ID'].tolist())
