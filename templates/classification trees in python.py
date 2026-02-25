#classification trees in python

#%%
import pandas as pd
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/web customer.csv")
df.head()
#-----------------------------------------------------------------------------
#%%
#divide into training and test data sets
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]
#-----------------------------------------------------------------------------
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
X=train[["site1","site2","site3","site4","site5"]]
y=train["customer"]
dt = DecisionTreeClassifier(random_state=17)
dt.fit(X, y)
Xtest=test[["site1","site2","site3","site4","site5"]]
ytest=test["customer"]
predictions = dt.predict(Xtest)
print(confusion_matrix(ytest, predictions))
#-----------------------------------------------------------------------------
#%%
#Visualizing the tree
from matplotlib import pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_= tree.plot_tree(dt,filled=True)
#-----------------------------------------------------------------------------
#%%
#make a prediction using the tree
newX= [[1,1,1,0,0]]
newX = pd.DataFrame(newX, columns=["site1", "site2", "site3", "site4", "site5"])
dt.predict(newX)
#predicts a 1, this user would be a customer
#newX= [[0,0,1,0,1]]
#newX = pd.DataFrame(newX, columns=["site1", "site2", "site3", "site4", "site5"])
#dt.predict(newX)
#predicts a 0, this user would not be a customer