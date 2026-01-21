# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:42:55 2025

@author: tyler
"""

#Multiplule Regressions Template
#Multiplule Regressions
import pandas as pd
mre = pd.read_csv("E:/Random Data/MultipleRealestate.csv")
mre.head()
#----------------------------------------------------------------------------
#%%
#divide into training and test data sets
import numpy as np
np.random.seed(42)
mre['r'] = np.random.uniform(size=len(mre))
#re.head()
train=mre[mre["r"] <= .6]
#train.head()
test=mre[mre["r"] > .6]
#test.head()
train=train.drop("r",axis=1)
test=test.drop("r",axis=1)
#----------------------------------------------------------------------------
#%%
#variable selection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
X = train[["Sqft","Location","Condition","Yard","NbrAvg","Economic"]]#Feature Matrix
y = train["SalePrice"]#Target Variable
X = preprocessing.scale(X)
X=pd.DataFrame(X, columns =["Sqft","Location","Condition","Yard","NbrAvg","Economic"])
estimator = LassoCV(cv=5)
sfm = SelectFromModel(estimator, prefit=False, max_features=None,threshold=0.25)
sfm.fit(X, y)
feature_idx = sfm.get_support()
selected_features = X.columns[feature_idx]
print(selected_features)
#----------------------------------------------------------------------------
#%%
#fit the model to the training data
import statsmodels.api as sm
X=train[selected_features]
X=sm.add_constant(X)
reg = sm.OLS(y,X).fit()
print(reg.summary())
#check to see that Prob(F-statistic)<0.05, to see if the model is significant
#----------------------------------------------------------------------------
#%%
#evaluate the model on the test data
tX=test[selected_features]
tX=sm.add_constant(tX)
py=reg.predict(tX)
ty=test["SalePrice"]
ae=abs(ty-py)
mae=np.mean(ae)
print("The mean absolute error is",mae)
N=len(test)
rmse=np.sqrt((np.sum((ty-py)**2))/N)
print("The root mean square error is",rmse)
#----------------------------------------------------------------------------
#%%
#make a prediction
#the first 1 is for the intercept
newX= [[1,1200,2,1,3,120000]]
reg.predict(newX)