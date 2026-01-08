#%%
#read the data
import sqlite3
import pandas as pd
import glob
import os
DB_PATH = "E:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM story", conn) 
print(df.head(2))
"""'Post ID', 'Account ID', 'Account username', 'Account name',
       'Description', 'Duration (sec)', 'Publish time', 'Permalink',
       'Post type', 'Data comment', 'Date', 'Views', 'Reach', 'Likes',
       'Shares', 'Profile visits', 'Replies', 'Sticker taps', 'Navigation',
       'Link clicks'"""
#----------------------------------------------------------------------------
#%%
#clean db
#if the values are null replace with 0
df=df.fillna(0)
#convert test columns to numeric data types
#----------------------------------------------------------------------------
#%%
#k-means
from sklearn.cluster import KMeans
tocluster=df[["Views","Profile visits"]]
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(tocluster)
#-----------------------------------------------------------------------------
#%%
#Apply CH to Different Numbers of Clusters:
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
tocluster=df[["Views","Profile visits"]]
for i in range(2,7):
  kmeans = KMeans(n_clusters=i,random_state=42)
  kmeans.fit(tocluster)
  ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
  print("Calinski–Harabasz Index for",i,"clusters:", ch_score)
#-----------------------------------------------------------------------------
#%%
#Continuing with 5 clusters
from sklearn.cluster import KMeans
tocluster=df[["Views","Profile visits"]]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(tocluster)
clusters = kmeans.predict(tocluster)
#-----------------------------------------------------------------------------
#%%
#Look at cluster membership
df['cluster'] = clusters.tolist()
print(df)
#-----------------------------------------------------------------------------
#%%
#cluster summary
p1=df.pivot_table(index="cluster",values=["Views","Profile visits"],aggfunc="mean")
p2=df.pivot_table(index="cluster",values=["Duration (sec)"],aggfunc="count")
print("Cluster Means \n",p1)
print("\n \n Number of Post type per Cluster \n",p2)
#-----------------------------------------------------------------------------
#%%