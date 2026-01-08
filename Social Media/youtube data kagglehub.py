#read the data
import pandas as pd
df = pd.read_csv("C:/Users/tyler/.cache/kagglehub/datasets/prince7489/youtube-shorts-performance-dataset/versions/1/youtube_shorts_performance_dataset.csv")
#%%
#------------------------------------------------------------------------------
#df column names
print(df.columns)
#['video_id', 'title', 'duration_sec', 'hashtags_count', 'views', 'likes','comments', 'shares', 'upload_hour', 'category']
#%%
#------------------------------------------------------------------------------
#descriptive stats
#number of post per category
num_post_cat = df['category'].value_counts().reset_index()
num_post_cat.columns = ['category', 'num_posts']

print(num_post_cat)

#plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(num_post_cat['category'], num_post_cat['num_posts'])

ax.set_title('Number of Posts per Category')
ax.set_xlabel('Category')
ax.set_ylabel('Number of Posts')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
#mean duration per category
average= (
    #df.groupby('to_group_by')['column_mess_with']
    df.groupby('category')[['duration_sec','views', 'likes','comments', 'shares']]
      .mean()   #how you want to math
      .reset_index()     #moves the index back into a normal column and creates a new DataFrame.
)
print(average)

#%%
#------------------------------------------------------------------------------
#-----------------------------------------
#heatmap for every upload hour 
#-----------------------------------------
#views
hourly_views = (
    df.groupby(['category','upload_hour'])['views']
      .mean()     # or .sum()
      .reset_index()
)

#plot
pivot = hourly_views.pivot(
    index='upload_hour',
    columns='category',
    values='views'
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title('Average Views by Upload Hour and Category views')
plt.ylabel('Upload Hour')
plt.xlabel('Category')
plt.show()

#likes
hourly_likes = (
    df.groupby(['category','upload_hour'])['likes']
      .mean()     # or .sum()
      .reset_index()
)

#plot
pivot = hourly_likes.pivot(
    index='upload_hour',
    columns='category',
    values='likes'
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title('Average Views by Upload Hour and Category likes')
plt.ylabel('Upload Hour')
plt.xlabel('Category')
plt.show()

#comments
hourly_comments = (
    df.groupby(['category','upload_hour'])['comments']
      .mean()     # or .sum()
      .reset_index()
)

#plot
pivot = hourly_comments.pivot(
    index='upload_hour',
    columns='category',
    values='comments'
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title('Average Views by Upload Hour and Category comments')
plt.ylabel('Upload Hour')
plt.xlabel('Category')
plt.show()

#shares
hourly_shares = (
    df.groupby(['category','upload_hour'])['shares']
      .mean()     # or .sum()
      .reset_index()
)

#plot
pivot = hourly_shares.pivot(
    index='upload_hour',
    columns='category',
    values='shares'
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title('Average Views by Upload Hour and Category shares')
plt.ylabel('Upload Hour')
plt.xlabel('Category')
plt.show()

#%%
#------------------------------------------------------------------------------
#what hour gets the most interactions
interactions=(
    df.groupby(['upload_hour'])[['likes','comments', 'shares']]
      .sum()     # or .sum()
      .reset_index()
)

interactions['total_interactions'] = (
    interactions['likes']
    + interactions['comments']
    + interactions['shares']
)


print(interactions)

#plot a line graph 
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(
    interactions['upload_hour'],
    interactions['total_interactions'],
    marker='o'
)

plt.title('Total Interactions by Upload Hour All')
plt.xlabel('Upload Hour')
plt.ylabel('Total Interactions')

plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#------------------------------------------------------------------------------
#what hour gets the most interactions (category)
interactions_category=(
    df.groupby(['upload_hour','category'])[['likes','comments', 'shares']]
      .sum()     # or .sum()
      .reset_index()
)

interactions_category['total_interactions'] = (
    interactions_category['likes']
    + interactions_category['comments']
    + interactions_category['shares']
)


print(interactions_category)

#plot a line graph 
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Loop over each category
for category in interactions_category['category'].unique():
    subset = interactions_category[interactions_category['category'] == category]
    plt.plot(
        subset['upload_hour'],
        subset['total_interactions'],
        marker='o',
        label=category  # this will show the category in the legend
    )

plt.title('Total Interactions by Upload Hour per Category')
plt.xlabel('Upload Hour')
plt.ylabel('Total Interactions')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.legend(title='Category')
plt.tight_layout()
plt.show()
#%%
#------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# Define columns
var1 = 'upload_hour'
var2 = 'likes'
constant = 'comments'  # lowercase to avoid NameError

# Optional: loop over each category
categories = df['category'].unique()

for cat in categories:
    print(f"\n--- Category: {cat} ---")
    df_cat = df[df['category'] == cat]
    
    # Prepare data for clustering
    tocluster = df_cat[[var1, var2]]
    
    # Optional: Determine best number of clusters using Calinski-Harabasz Index
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(tocluster)
        ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
        print(f"CH Index for {i} clusters: {ch_score:.2f}")
    
    # Fit KMeans with chosen number of clusters (5 here)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(tocluster)
    
    # Add cluster labels to DataFrame
    df_cat = df_cat.copy()
    df_cat['cluster'] = clusters
    
    # Cluster summary: mean of likes and views
    p1 = df_cat.pivot_table(index='cluster', values=[var1, var2], aggfunc='mean')
    
    # Cluster summary: count of comments
    p2 = df_cat.pivot_table(index='cluster', values=[constant], aggfunc='count')
    
    print("\nCluster Means:\n", p1)
    print(f"\nNumber of {constant} per Cluster:\n", p2)
    
    # Plot clusters
    plt.figure(figsize=(8, 5))
    sns.set_style('white')
    sns.set_palette(['red', 'blue', 'green', 'yellow', 'black'])
    
    sns.scatterplot(
        data=df_cat,
        x=var1,
        y=var2,
        hue='cluster',
        palette='Set1',
        s=60
    )
    
    plt.title(f'K-Means Clusters for Category: {cat}')
    plt.xlabel(var1.capitalize())
    plt.ylabel(var2.capitalize())
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

print("""
      tech focus on cluster 1 most likes around 10:25 post
      comedy focus on cluster 1 and 4 likes 1: 2524.600000 best post time is 17:36 4:36982.000000     9:51
    food 2 and 4 2:44179.181818    12:11 4: 36438.909091     8:380
    lifestyle 3 45079.625       12:38
    travel 1 43787.285714    14:17
    education 0 and 4 0: 47156.428571    10:000000   4:39806.818182     8:16""")

#%%
#------------------------------------------------------------------------------
#random forrest to predict views per category 
independent_varaibles=['duration_sec', 'hashtags_count','likes','comments', 'shares', 'upload_hour']
dependent_variables="views"

#Divide data into train and test sets
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]

#Random Forest in Python
from sklearn.ensemble import RandomForestRegressor
X=train[independent_varaibles]
y=train[dependent_variables]
Xtest=test[independent_varaibles]
ytest=test[dependent_variables]
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=17
)
rf.fit(X,y)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preds = rf.predict(Xtest)

print(f"MAE (average error): ±{mean_absolute_error(ytest, preds):,.0f} views")
print(f"RMSE: {np.sqrt(mean_squared_error(ytest, preds)):,.0f} views")
print(f"R² (closer to 1 is better): {r2_score(ytest, preds):.3f}")

#%%
#------------------------------------------------------------------------------