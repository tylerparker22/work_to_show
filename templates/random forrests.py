#random forrests
#-----------------------------------------------------------------------------
#%%
#import dataset
import pandas as pd
test1 = pd.read_csv('https://raw.githubusercontent.com/blacktreeM/econ/refs/heads/main/test.csv')
test1['year'] = test1['datetime'].str[0:4]  # extract first 4 characters to make year column
test1['month'] = test1['datetime'].str[5:7]  # extract characters 5:7
test1['hour'] = test1['datetime'].str[11:13]  # extract characters 11:13
test1['day'] = pd.to_datetime(test1['datetime']).dt.day_name()
df=test1
df.head()
#-----------------------------------------------------------------------------
#%%
VALUES=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'hour', 'day']
TARGET_COL="datetime"
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
X=train[VALUES]
y=train[TARGET_COL]
Xtest=test[VALUES]
ytest=test[TARGET_COL]
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