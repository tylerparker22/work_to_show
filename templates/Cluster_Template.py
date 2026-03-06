#%%
#read the data
import pandas as pd
df = pd.read_csv("//sdata/susers/tparker27/My Documents/GitHub/work_to_show/Classes Spring 2026/BDA Spring 26/Datasets/Pie ingredients.csv")
df.head()
df.columns # ['butter', 'oil', 'rating']
#-----------------------------------------------------------------------------
#%%
var1 = 'Exam'
var2 = 'Homework'
Constant = 'Student'

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
print("Cluster Means /n",p1)
print("/n /n Number of, {Constant}, per Cluster /n",p2)

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

# %%
