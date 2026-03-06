# 3.13.7 python works
# %% import data
import pandas as pd
df=pd.read_csv("//sdata/susers/tparker27/My Documents/GitHub/work_to_show/Sports/Datasets SA Class/SBR001-715.csv")
#----------------------------

# %% column names
print(df.columns.tolist())
# ['Item', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', np.float64(nan)]
#----------------------------
# %% summary stats
# columns to clean
cols = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']

# 1️⃣ remove $ and commas + convert to numeric
df[cols] = (
    df[cols]
    .replace(r'[/$,]', '', regex=True)
    .apply(pd.to_numeric, errors='coerce')
)

# 2️⃣ now drop NA rows
df = df.dropna(subset=['Item'] + cols)

# 3️⃣ summary stats
df.describe()

#----------------------------------

# %% Items unique
itme_unqiue=df['Item'].unique()
itme_unqiue

['Item',
       'Total Number of Fans Age 13+ (View and/or Attend - add ,000)  ',
       'Total Number Viewing on TV (add 000)',
       '% of TV Viewers who Viewed More Than 2-5/yr.',
       '% of TV Viewers who Viewed 6+ Times/yr.', 'All Fans',
       'Viewed at Least One Minor League Baseball Game On TV',
       'Viewed 6+ Minor League Baseball Games on TV',
       'Followed Minor League Baseball On Facebook',
       'Followed Minor League Baseball On X (formerly Twitter)',
       'Purchased Minor League Baseball Logo Apparel',
       'Follow Minor League Baseball on Facebook - Total all Followers (add 000) ',
       'Follow Minor League Baseball on X (formerly Twitter) - Total all Followers (add 000)',
       'Expenditures for Sports Logo Apparel (add 000)',
       '% of All Fans Purchased Sports Logo Apparel',
       'Base (Total No. of Minor League Baseball Fans - add ,000)',
       '% Saying Sport Sponsorship is Extremely Influential',
       '% Saying Sport Sponsorship is Very Influential',
       '% Saying Sport Sponsorship is Moderately Influential',
       '% Saying Sport Sponsorship is Slightly Influential',
       '% Saying Sport Sponsorship is Not at All Influential', 'Total']
# %% regression model
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import summary_table

# Set 'Item' as the index and transpose so items become columns
df_transposed = df.set_index('Item').T

# Clean up - remove the first row if it's a duplicate header
if df_transposed.index[0] == 'Item':
    df_transposed = df_transposed.iloc[1:]

# Convert to numeric - use apply instead of a loop
df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')

# Check the structure
print("Transposed data shape:", df_transposed.shape)
print("/nColumn names (first 10):")
print(df_transposed.columns[:10].tolist())
print("/nFirst few rows:")
print(df_transposed.head())

# Now run your regression with the transposed data
var1 = 'Viewed at Least One Minor League Baseball Game On TV'
var2 = 'Purchased Minor League Baseball Logo Apparel'

# Check if columns exist
print(f"/n{var1} in columns: {var1 in df_transposed.columns}")
print(f"{var2} in columns: {var2 in df_transposed.columns}")

# Split to training and test datasets
np.random.seed(42)
df_transposed['r'] = np.random.uniform(size=len(df_transposed))

train = df_transposed[df_transposed["r"] <= .6]
test = df_transposed[df_transposed["r"] > .6]

train = train.drop("r", axis=1)
test = test.drop("r", axis=1)

# Regression on training data
y = train[var2]
x = train[var1]
x = sm.add_constant(x)
reg = sm.OLS(y, x).fit()
print("/n" + "="*60)
print("REGRESSION SUMMARY")
print("="*60)
print(reg.summary())

# Mean absolute error (MAE) for test data
tx = test[var1]
tx = sm.add_constant(tx)
psales = reg.predict(tx)
tsales = test[var2]
ae = abs(tsales - psales)
mae = np.mean(ae)
print(f"/nThe mean absolute error is {mae:.4f}")

# Root Mean Square Error
N = len(test)
rmse = np.sqrt((np.sum((tsales - psales)**2)) / N)
print(f"The root mean square error is {rmse:.4f}")

# Prediction interval plot
x = train[var1]
st, data, ss2 = summary_table(reg, alpha=0.05)
fittedvalues = data[:, 2]
predict_ci_low, predict_ci_upp = data[:, 6:8].T

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'o', label='Actual', markersize=8)
ax.plot(x, fittedvalues, '-', lw=2, label='Fitted', color='blue')
ax.plot(x, predict_ci_low, 'r--', lw=2, label='95% Prediction Interval')
ax.plot(x, predict_ci_upp, 'r--', lw=2)
ax.set_title(f"Predicting {var2}/nfrom {var1}", fontsize=12)
ax.set_xlabel(var1, fontsize=10)
ax.set_ylabel(var2, fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %% what is the mean hh income if they attended one game? 
varibale='Viewed at Least One Minor League Baseball Game On TV'

