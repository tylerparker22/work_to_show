"""In the file "student success". the columns are as follows: FinalGrade1403 is 
the final grade in a first class in computing (on a four point scale), 
Transfer (1 if they transferred from another school, 0 if not), 
ACTMath is the student's score on the math portion of the ACT exam, HSGPA is 
the student's high school GPA, success is 1 for successfully completed a 
computing degree, 0 for did not finish.

Using a classification tree, a random forest, and an artificial neural network in python:

Divide the data into training and test data sets. Predict student success in a 
computing degree based on all of the provided data, assume variable selection
 has already been completed. (In other words, predict the column target_variable 
using the columns: "FinalGrade1403", "Transfer", "ACTMath", and "HSGPA" as 
predictors.) Create a confusion matrix for the test data, calculate accuracy 
from the confusion matrix and make a prediction for a new student. Compare the 
confusion matrices for all three techniques, which did the best? According to 
random forest, which variables are most important to predicting student 
success? 
Turn in your python code, the accuracy for each method, 
the variable importance, and your recommendation for which of the 
three models to use."""
#------------------------------------------------------------------------------
#%%
#import dataset
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/student success.csv")
#-----------------------------------------------------------------------------
#%%
#split data into train and test data
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train=df[df["r"] <= .6]
test=df[df["r"] > .6]
#identify predictors and target columns in train and test data
X=train[["FinalGrade1403","Transfer","ACTMath","HSGPA"]]
y=train["success"]
Xtest=test[["FinalGrade1403","Transfer","ACTMath","HSGPA"]]
ytest=test["success"]
rf = RandomForestClassifier(criterion='gini',n_estimators=500,random_state=17)
rf.fit(X,y)
dt = DecisionTreeClassifier(random_state=17)
dt.fit(X, y)
#-----------------------------------------------------------------------------
#%%
#ANN
#Build the ANN and Create a Confusion Matrix for the Test Data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
clf=MLPClassifier(solver='lbfgs',alpha=0.00001,hidden_layer_sizes=(4,2),random_state=4)
clf.fit(X, y)
predictions = clf.predict(Xtest)
print(confusion_matrix(ytest, predictions))
newX= [[3,0,20,4]]
clf.predict(newX)
"""[[  0 161]
 [  0 656]] Out[37]: array([1])"""
#-----------------------------------------------------------------------------
#%%
#Classification Tree
#Visualizing the tree
predictions = dt.predict(Xtest)
print(confusion_matrix(ytest, predictions))
from matplotlib import pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(25,20))
temp= tree.plot_tree(dt,filled=True)
newX= [[3,0,20,4]]
dt.predict(newX)

"""[[ 85  76]
 [ 56 600]]

Out[38]: array([0])"""
#---------------------------------------------------------------
#%%
#random forrest
#predict
rf = RandomForestClassifier(criterion='gini',n_estimators=500,random_state=17)
rf.fit(X,y)
predictions = rf.predict(Xtest)
print(confusion_matrix(ytest, predictions))
newX= [[3,0,20,4]]
rf.predict(newX)
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X.columns[0:]
for f in range(X.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 30,feat_labels[sorted_indices[f]],importances[sorted_indices[f]]))
from matplotlib import pyplot as plt
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

"""
[[ 94  67]
 [ 32 624]]

 1) FinalGrade1403                 0.385531
 2) ACTMath                        0.340772
 3) HSGPA                          0.256641
 4) Transfer                       0.017056
"""