#random forrests
#-----------------------------------------------------------------------------
#%%
#import dataset
import pandas as pd
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/web customer.csv")
df.head()
#-----------------------------------------------------------------------------
#%%
#Divide data into train and test sets
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]
#-----------------------------------------------------------------------------
#%%
#Random Forest in Python
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X=train[["site1","site2","site3","site4","site5"]]
y=train["customer"]
Xtest=test[["site1","site2","site3","site4","site5"]]
ytest=test["customer"]
rf = RandomForestClassifier(criterion='gini',n_estimators=500,random_state=17)
rf.fit(X,y)
#-----------------------------------------------------------------------------
#%%
#predict
predictions = rf.predict(Xtest)
print(confusion_matrix(ytest, predictions))
#-----------------------------------------------------------------------------
#%%
#Make a prediction using the random forest
newX= [[1,1,1,0,0]]
rf.predict(newX)
#-----------------------------------------------------------------------------
#%%
#Additional Benefit of Random Forest Models: Feature Selection
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X.columns[0:]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[sorted_indices[f]],importances[sorted_indices[f]]))
#-----------------------------------------------------------------------------
#%%
#Feature Importance Graphed
from matplotlib import pyplot as plt
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()