import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob

# Load dataset
df = pd.read_csv("D:/Python/Random Projects In Python/Instagram data.csv")

# %% text features

#caption
df["Caption"] = df["Caption"].astype(str)
df["sentiment"] = df["Caption"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["Caption"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["len_chars"] = df["Caption"].apply(len)
df["len_words"] = df["Caption"].apply(lambda x: len(x.split()))

#Hashtags
df["Hashtags"] = df["Hashtags"].astype(str)
df["sentiment"] = df["Hashtags"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["Hashtags"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["len_chars"] = df["Hashtags"].apply(len)
df["len_words"] = df["Hashtags"].apply(lambda x: len(x.split()))

# %% print columns
print(df.columns)
"""['Impressions', 'From Home', 'From Hashtags', 'From Explore',
       'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits',
       'Follows', 'Caption', 'Hashtags', 'sentiment', 'subjectivity',
       'len_chars', 'len_words']"""

# %%average per column
numeric_cols=['Impressions', 'From Home', 'From Hashtags', 'From Explore',
       'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits',
       'Follows', 'sentiment', 'subjectivity',
       'len_chars', 'len_words']

# data frame for nicer display
col_means = round(df[numeric_cols].mean(),2).reset_index()
col_means.columns = ['Metric','Average']
col_means

"""
            Metric  Average
0      Impressions  5703.99
1        From Home  2475.79
2    From Hashtags  1887.51
3     From Explore  1078.10
4       From Other   171.09
5            Saves   153.31
6         Comments     6.66
7           Shares     9.36
8            Likes   173.78
9   Profile Visits    50.62
10         Follows    20.76
11       sentiment     0.01
12    subjectivity     0.00
13       len_chars   264.81
14       len_words    18.97
"""
# %% feature importance

"""
most important for impressions is too have good hashtags
most important for likes is too be 'from explorer'
"""

var1=['Impressions','From Home', 'From Hashtags', 'From Explore',
       'From Other', 'Saves', 'Comments', 'Shares',  'Profile Visits',
       'Follows', 'sentiment', 'subjectivity',
       'len_chars', 'len_words']
var2='Likes'

#Divide data into train and test sets
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]


#Random Forest in Python
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X=train[var1]
y=train[var2]
Xtest=test[var1]
ytest=test[var2]
rf = RandomForestClassifier(criterion='gini',n_estimators=500,random_state=17)
rf.fit(X,y)


#predict
predictions = rf.predict(Xtest)
print(confusion_matrix(ytest, predictions))


#Additional Benefit of Random Forest Models: Feature Selection
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X.columns[0:]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[sorted_indices[f]],importances[sorted_indices[f]]))


#Feature Importance Graphed
from matplotlib import pyplot as plt
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

# %% likes from impressions cluster

var1 = 'Likes'
var2 = 'Impressions'
Constant = 'len_chars'

#k-means
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(tocluster)

#Apply CH to Different Numbers of Clusters:
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
for i in range(2,7):
  kmeans = KMeans(n_clusters=i,random_state=42)
  kmeans.fit(tocluster)
  ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
  print("Calinski–Harabasz Index for",i,"clusters:", ch_score)
  
#Continuing with 5 clusters
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(tocluster)
clusters = kmeans.predict(tocluster)

#Look at cluster membership
df['cluster'] = clusters.tolist()
print(df)

#cluster summary
p1=df.pivot_table(index="cluster",values=[var1,var2],aggfunc="mean")
p2=df.pivot_table(index="cluster",values=[Constant],aggfunc="count")
print("Cluster Means \n",p1)
print("\n \n Number of, {Constant}, per Cluster \n",p2)

#plotting clusters
import seaborn as sns
import matplotlib.pyplot as plt
df["Clusters"]=clusters
#set font size of labels on matplotlib plots
plt.rc('font', size=8)
sns.set_style('white')
#set the colors used in the graph, see https://xkcd.com/color/rgb/
customPalette = ['red', 'blue', 'green', 'yellow', 'black' ]
sns.set_palette(customPalette)
#plot data with seaborn
facet = sns.lmplot(data=df, x=var1, y=var2, hue='Clusters',
fit_reg=False, legend=True)

# %% cluster of 'sentiment', 'subjectivity'
var1 = 'Profile Visits'
var2 = 'Likes'
Constant = 'sentiment'

#k-means
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(tocluster)

#Apply CH to Different Numbers of Clusters:
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
for i in range(2,7):
  kmeans = KMeans(n_clusters=i,random_state=42)
  kmeans.fit(tocluster)
  ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
  print("Calinski–Harabasz Index for",i,"clusters:", ch_score)
  
#Continuing with 5 clusters
from sklearn.cluster import KMeans
tocluster=df[[var1,var2]]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(tocluster)
clusters = kmeans.predict(tocluster)

#Look at cluster membership
df['cluster'] = clusters.tolist()
print(df)

#cluster summary
p1=df.pivot_table(index="cluster",values=[var1,var2],aggfunc="mean")
p2=df.pivot_table(index="cluster",values=[Constant],aggfunc="count")
print("Cluster Means \n",p1)
print("\n \n Number of, {Constant}, per Cluster \n",p2)

#plotting clusters
import seaborn as sns
import matplotlib.pyplot as plt
df["Clusters"]=clusters
#set font size of labels on matplotlib plots
plt.rc('font', size=8)
sns.set_style('white')
#set the colors used in the graph, see https://xkcd.com/color/rgb/
customPalette = ['red', 'blue', 'green', 'yellow', 'black' ]
sns.set_palette(customPalette)
#plot data with seaborn
facet = sns.lmplot(data=df, x=var1, y=var2, hue='Clusters',
fit_reg=False, legend=True)

# %% Cluster text columns for posts with Likes above mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Threshold: Likes above average
mean_likes = df['Likes'].mean()
df_above = df[df['Likes'] > mean_likes].copy()  # only keep posts above mean

text_cols = ['Caption', 'Hashtags']

# Combine the text columns into one for clustering
df_above['combined_text'] = df_above[text_cols].apply(lambda x: ' '.join(x), axis=1)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_text = vectorizer.fit_transform(df_above['combined_text'])

# Apply KMeans clustering
n_clusters = 5  # you can change this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_text)

# Add cluster labels to the DataFrame
df_above['text_cluster'] = clusters

# Summary: number of posts per cluster
cluster_summary = df_above.groupby('text_cluster').size().reset_index(name='post_count')
print("Cluster Summary:\n", cluster_summary)

# Optional: top terms per cluster
terms = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_idx = cluster_center.argsort()[::-1][:10]  # top 10 words
    top_terms = [terms[idx] for idx in top_terms_idx]
    print(f"Cluster {i} top words:", top_terms)

# %% Random Forest to predict Impressions (numeric target)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Features and target
var1 = [ 'From Home', 'From Hashtags', 'From Explore',
         'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits',
         'Follows', 'sentiment', 'subjectivity',
         'len_chars', 'len_words']
var2 = 'Impressions'

# Train/test split
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train = df[df['r'] <= 0.6]
test  = df[df['r'] > 0.6]

X = train[var1]
y = train[var2]
Xtest = test[var1]
ytest = test[var2]

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=500, random_state=17)
rf.fit(X, y)

# Predict on test set
predictions = rf.predict(Xtest)
mae = mean_absolute_error(ytest, predictions)
rmse = np.sqrt(mean_squared_error(ytest, predictions))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Predict for a new example
newX = [[
    2476, # From Home 
    1888, # From Hashtags
    1078, # From Explore
    171,  # From Other
    153,  # Saves
    7,    # Comments
    9,    # Shares
    174,  # Likes
    51,   # Profile Visits
    21,   # Follows
    0,    # sentiment
    0,    # subjectivity
    265,  # len_chars
    19    # len_words
]]
predicted_impressions = rf.predict(newX)
print("Predicted Impressions for new post:", round(predicted_impressions[0], 2))


# %% to do list

1. random forrest to predict a post columns