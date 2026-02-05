#Artificial Neural Networks
#-----------------------------------------------------------------------------
#%%
#import dataset
import pandas as pd
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/web customer.csv")
df.head()
#-----------------------------------------------------------------------------
#%%
#split data into train and test data
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]
#identify predictors and target columns in train and test data
X=train[["site1","site2","site3","site4","site5"]]
y=train["customer"]
Xtest=test[["site1","site2","site3","site4","site5"]]
ytest=test["customer"]
#-----------------------------------------------------------------------------
#%%
#Build the ANN and Create a Confusion Matrix for the Test Data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
clf=MLPClassifier(solver='lbfgs',alpha=0.00001,hidden_layer_sizes=(4,2),random_state=4)
clf.fit(X, y)
predictions = clf.predict(Xtest)
print(confusion_matrix(ytest, predictions))
#-----------------------------------------------------------------------------
#%%
#Make a prediction using ANN
#[[1, "yes", "yes", "no","no"]]
newX= [[1,1,1,0,0]]
clf.predict(newX)
#-----------------------------------------------------------------------------
#%%