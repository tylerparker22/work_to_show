#random forrests

#import dataset
import pandas as pd
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/web customer.csv")
df.head()
# =====================================================================================================
# RandomForestClassifier
VALUES = ["site1","site2","site3","site4","site5"]          # list of input feature column names
TARGET_COL = "customer"                                      # column we are trying to predict

# Divide data into train and test sets
import numpy as np
np.random.seed(42)                                           # set seed so random results are reproducible
df['r'] = np.random.uniform(size=len(df))                   # add column of random numbers (0 to 1) to each row
train = df[df["r"] <= .6]                                    # ~60% of rows go to training set
test = df[df["r"] > .6]                                      # ~40% of rows go to test set

# Random Forest in Python
from sklearn.metrics import confusion_matrix                  # import tool to evaluate predictions
from sklearn.ensemble import RandomForestClassifier          # import the random forest model

X = train[VALUES]                                            # training features (inputs)
y = train[TARGET_COL]                                        # training target (what we predict)
Xtest = test[VALUES]                                         # test features (inputs)
ytest = test[TARGET_COL]                                     # test target (what we predict)

rf = RandomForestClassifier(criterion='gini',                # use gini impurity to measure split quality
                             n_estimators=500,               # build 500 decision trees
                             random_state=17)                # seed for reproducibility
rf.fit(X, y)                                                 # train the model on training data

# Predict
predictions = rf.predict(Xtest)                              # use trained model to predict on test set
print(confusion_matrix(ytest, predictions))                  # print matrix comparing predictions vs actual

# Make a prediction using the random forest
newX = [[1,1,1,0,0]]                                         # one new data point with 5 feature values
rf.predict(newX)                                             # predict the customer class for this new row

# Additional Benefit of Random Forest Models: Feature Selection
importances = rf.feature_importances_                        # get importance score for each feature
sorted_indices = np.argsort(importances)[::-1]               # sort indices from most to least important
feat_labels = X.columns[0:]                                  # grab feature column names

for f in range(X.shape[1]):                                  # loop through each feature
    print("%2d) %-*s %f" % (f + 1, 30,                      # print rank number
                             feat_labels[sorted_indices[f]], # print feature name
                             importances[sorted_indices[f]]))# print importance score

# Feature Importance Graphed
from matplotlib import pyplot as plt
plt.title('Feature Importance')                              # set chart title
plt.bar(range(X.shape[1]),                                   # create bar for each feature
        importances[sorted_indices], align='center')         # bar height = importance score
plt.xticks(range(X.shape[1]),                                # set x-axis tick positions
           X.columns[sorted_indices], rotation=90)           # label ticks with feature names, rotated
plt.tight_layout()                                           # adjust layout so labels don't get cut off
plt.show()                                                   # display the chart
# =====================================================================================================
VALUES = ["site1","site2","site3","site4","site5"]           # list of input feature column names
TARGET_COL = "customer"                                      # column we are trying to predict

# Divide data into train and test sets
import numpy as np
np.random.seed(42)                                           # set seed so random results are reproducible
df['r'] = np.random.uniform(size=len(df))                   # add column of random numbers (0 to 1) to each row
train = df[df["r"] <= .6]                                    # ~60% of rows go to training set
test = df[df["r"] > .6]                                      # ~40% of rows go to test set

# Random Forest in Python
from sklearn.metrics import mean_squared_error, r2_score     # regression evaluation metrics
from sklearn.ensemble import RandomForestRegressor           # regressor instead of classifier

X = train[VALUES]                                            # training features (inputs)
y = train[TARGET_COL]                                        # training target (what we predict)
Xtest = test[VALUES]                                         # test features (inputs)
ytest = test[TARGET_COL]                                     # test target (what we predict)

rf = RandomForestRegressor(n_estimators=500,                 # build 500 decision trees
                            random_state=17)                 # seed for reproducibility
rf.fit(X, y)                                                 # train the model on training data

# Predict
predictions = rf.predict(Xtest)                              # use trained model to predict on test set
print("MSE: ", mean_squared_error(ytest, predictions))       # avg squared error (lower is better)
print("R²:  ", r2_score(ytest, predictions))                 # 1.0 = perfect fit, 0 = bad fit

# Make a prediction using the random forest
newX = [[1,1,1,0,0]]                                         # one new data point with 5 feature values
print(rf.predict(newX))                                      # predict the numeric value for this new row

# Additional Benefit of Random Forest Models: Feature Selection
importances = rf.feature_importances_                        # get importance score for each feature
sorted_indices = np.argsort(importances)[::-1]               # sort indices from most to least important
feat_labels = X.columns[0:]                                  # grab feature column names

for f in range(X.shape[1]):                                  # loop through each feature
    print("%2d) %-*s %f" % (f + 1, 30,                      # print rank number
                             feat_labels[sorted_indices[f]], # print feature name
                             importances[sorted_indices[f]]))# print importance score

# Feature Importance Graphed
from matplotlib import pyplot as plt
plt.title('Feature Importance')                              # set chart title
plt.bar(range(X.shape[1]),                                   # create bar for each feature
        importances[sorted_indices], align='center')         # bar height = importance score
plt.xticks(range(X.shape[1]),                                # set x-axis tick positions
           X.columns[sorted_indices], rotation=90)           # label ticks with feature names, rotated
plt.tight_layout()                                           # adjust layout so labels don't get cut off
plt.show()                                                   # display the chart
# =====================================================================================================