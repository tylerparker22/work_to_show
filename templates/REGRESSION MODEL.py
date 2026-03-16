#REGRESSION MODEL
#READ DATASET
import pandas as pd
df = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/realestate.csv")
print(df.columns.tolist())
#----------------------------------------------------------------------------
#%%
#split to training and test datasets
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
#df.head()
train=df[df["r"] <= .6]
#train.head()
test=df[df["r"] > .6]
#test.head()
train=train.drop("r",axis=1)
test=test.drop("r",axis=1)

#regression on training data
import statsmodels.api as sm
import matplotlib.pyplot as plt
y=train[var2]
x=train[var1]
x = sm.add_constant(x) #needed to force an intercept into the model
reg = sm.OLS(y,x).fit()
print(reg.summary())

#mean of absolute error (MAE) for test data
tx=test[var1]
tx=sm.add_constant(tx)
psales=reg.predict(tx)
tsales=test[var2]
ae=abs(tsales-psales)
mae=np.mean(ae)
print("The mean absolute error is",mae)

#Root Mean Square Error of test predictions
N=len(test)
rmse=np.sqrt((np.sum((tsales-psales)**2))/N)
print("The root mean square error is",rmse)

#Prediction interval plot
#-------------------
#remove the intercept for the scatterplot
x=train[var1]
#import another regression module
from statsmodels.stats.outliers_influence import summary_table
#create a table of regression information from our regression problem
st, data, ss2 = summary_table(reg, alpha=0.05)
#extract the predicted values to create the regression line on the graph
fittedvalues = data[:, 2]
#extract the confidence intervals for an individual value (prediction intervals)
predict_ci_low, predict_ci_upp = data[:, 6:8].T
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
temp=ax.plot(x, y, 'o')
temp=ax.plot(x, fittedvalues, '-', lw=2)
temp=ax.plot(x, predict_ci_low, 'r--', lw=2)
temp=ax.plot(x, predict_ci_upp, 'r--', lw=2)
temp=ax.set_title(f"Predicting {var2} from {var1}")
temp=ax.set_xlabel(var1)
temp=ax.set_ylabel(var2)
