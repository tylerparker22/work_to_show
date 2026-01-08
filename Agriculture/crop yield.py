# %% crop yeild prediciton
import pandas as pd

df=pd.read_csv("C:/Users/tyler/Downloads/crop-yield.csv")
# %% column names 
print(df.columns)
"""
['N', 'P', 'K', 'Soil_pH', 'Soil_Moisture', 'Soil_Type',
       'Organic_Carbon', 'Temperature', 'Humidity', 'Rainfall',
       'Sunlight_Hours', 'Wind_Speed', 'Region', 'Altitude', 'Season',
       'Crop_Type', 'Irrigation_Type', 'Fertilizer_Used', 'Pesticide_Used',
       'Crop_Yield_ton_per_hectare']
"""
# %% what crop produced the most yeilds? 
crop_yields = (
    df.groupby(['Crop_Type'], as_index=False)
      ['Crop_Yield_ton_per_hectare']
      .sum()
      .sort_values(by='Crop_Yield_ton_per_hectare', ascending=False)
)

crop_yields

# """
#    Crop_Type  Crop_Yield_ton_per_hectare
# 4  sugarcane                   122197.81
# 2     potato                    41567.99
# 1      maize                    16706.07
# 3       rice                    16002.25
# 5      wheat                    14578.94
# 0     cotton                    12374.95
# """

# """ 
# sugarcane can be grown everywhere. Such an easy crop to grow. 
# potato can be grown pretty much everywhere 
# maize:"corn" can be grown pretty much everywhere not a hard crop
# rice: very specific about where it wants to grow, easiest to produce mass amounts if in water
# wheat: winter crop
# cotton: sandier crop loves sand dirt
# """
# %% crop sum by region
crop_region = (
    df.groupby(['Crop_Type', 'Region'], as_index=False)
      ['Crop_Yield_ton_per_hectare']
      .sum()
      .sort_values(by='Crop_Yield_ton_per_hectare', ascending=False)
)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(crop_region.to_string(index=False))
# '''
# Crop_Type  Region  Crop_Yield_ton_per_hectare
# sugarcane    west                    25900.69
# sugarcane   south                    25238.99
# sugarcane    east                    24754.41
# sugarcane central                    24601.05
# sugarcane   north                    21702.67
#    potato    west                     8639.85
#    potato    east                     8357.63
#    potato central                     8348.68
#    potato   south                     8114.10
#    potato   north                     8107.73
#     maize   south                     3586.37
#     maize   north                     3504.97
#     maize    west                     3357.26
#      rice   north                     3356.05
#      rice    west                     3248.41
#      rice central                     3230.10
#     wheat    east                     3182.92
#      rice   south                     3154.28
#     maize    east                     3132.66
#     maize central                     3124.81
#      rice    east                     3013.41
#     wheat   south                     2971.50
#     wheat   north                     2952.70
#     wheat central                     2791.35
#     wheat    west                     2680.47
#    cotton   north                     2587.67
#    cotton central                     2580.21
#    cotton    east                     2563.96
#    cotton   south                     2362.19
#    cotton    west                     2280.92

# '''
# %% what crop grows the best in what region and season? 


crop_region_season = (
    df.groupby(['Crop_Type', 'Region','Soil_Type'], as_index=False)
      ['Crop_Yield_ton_per_hectare']
      .sum()
      .sort_values(by='Crop_Yield_ton_per_hectare', ascending=False)
)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(crop_region_season.to_string(index=False))


# """
# Crop_Type  Region Soil_Type  Crop_Yield_ton_per_hectare
# Sugarcane    West      Clay                     8174.54
# Sugarcane    East      Silt                     7477.03
# Sugarcane Central      Silt                     7109.36
# Sugarcane   South      Silt                     6668.75
# Sugarcane   South     Sandy                     6623.30
# Sugarcane    West     Loamy                     6598.45
# Sugarcane    West      Silt                     6586.08
# Sugarcane Central     Loamy                     6431.91
# Sugarcane   North     Loamy                     6273.17
# Sugarcane   South     Loamy                     6191.09
# Sugarcane   North      Clay                     6072.10
# Sugarcane    East     Loamy                     5980.06
# Sugarcane   South      Clay                     5755.85
# Sugarcane    East      Clay                     5697.11
# Sugarcane    East     Sandy                     5600.21
# Sugarcane Central      Clay                     5543.09
# Sugarcane Central     Sandy                     5516.69
# Sugarcane   North     Sandy                     5020.01
# Sugarcane    West     Sandy                     4541.62
# Sugarcane   North      Silt                     4337.39
#    Potato    East     Loamy                     2459.97
#    Potato    West     Loamy                     2327.43
#    Potato Central      Clay                     2263.01
#    Potato   South     Sandy                     2190.32
#    Potato    West     Sandy                     2190.25
#    Potato    West      Silt                     2162.49
#    Potato   South     Loamy                     2135.74
#    Potato    East     Sandy                     2117.57
#    Potato Central     Sandy                     2112.53
#    Potato   North      Clay                     2101.78
#    Potato   North     Loamy                     2088.87
#    Potato Central      Silt                     2069.05
#    Potato    East      Clay                     2051.19
#    Potato   South      Silt                     2041.71
#    Potato   North      Silt                     1994.26
#    Potato    West      Clay                     1959.68
#    Potato   North     Sandy                     1922.82
#    Potato Central     Loamy                     1904.09
#    Potato   South      Clay                     1746.33
#    Potato    East      Silt                     1728.90
#     Maize   South     Sandy                     1013.76
#     Maize    West     Sandy                      976.15
#      Rice    West     Loamy                      928.37
#     Maize   South      Clay                      927.74
#      Rice   North     Loamy                      920.93
#     Maize   North      Silt                      906.89
#     Maize   North     Loamy                      905.91
#      Rice   North     Sandy                      901.36
#     Maize   South      Silt                      899.84
#     Maize    East     Sandy                      890.24
#     Wheat   South      Silt                      876.43
#     Maize Central      Clay                      875.37
#      Rice Central     Loamy                      870.21
#     Maize   North      Clay                      863.83
#     Maize    West      Silt                      850.83
#      Rice   South     Loamy                      849.74
#      Rice    West      Clay                      849.54
#     Wheat    East     Loamy                      842.94
#    Cotton    East     Sandy                      839.05
#     Maize   North     Sandy                      828.34
#     Maize Central     Sandy                      824.19
#      Rice Central     Sandy                      823.51
#      Rice    East     Loamy                      817.35
#     Wheat    East     Sandy                      808.88
#      Rice   South      Clay                      807.12
#     Wheat   North      Clay                      806.15
#     Maize    East      Silt                      794.84
#     Wheat   North      Silt                      793.81
#     Wheat Central      Clay                      784.38
#      Rice Central      Silt                      782.89
#     Wheat    East      Silt                      781.66
#      Rice   North      Silt                      781.59
#     Maize    West      Clay                      778.86
#      Rice   South     Sandy                      774.91
#      Rice    East      Silt                      772.37
#     Wheat    West     Sandy                      762.16
#     Wheat   South     Loamy                      761.97
#     Maize    East     Loamy                      755.06
#      Rice Central      Clay                      753.49
#      Rice   North      Clay                      752.17
#     Maize    West     Loamy                      751.42
#     Wheat    East      Clay                      749.44
#      Rice    West      Silt                      747.95
#     Maize   South     Loamy                      745.03
#      Rice    West     Sandy                      722.55
#      Rice   South      Silt                      722.51
#      Rice    East     Sandy                      721.95
#     Wheat   South      Clay                      721.28
#     Maize Central      Silt                      718.47
#    Cotton   North     Loamy                      717.85
#    Cotton   North     Sandy                      712.26
#     Maize Central     Loamy                      706.78
#     Wheat Central     Loamy                      702.14
#      Rice    East      Clay                      701.74
#     Wheat    West     Loamy                      700.70
#     Wheat   North     Loamy                      698.51
#     Maize    East      Clay                      692.52
#    Cotton Central     Loamy                      689.31
#    Cotton    West      Silt                      687.74
#     Wheat Central      Silt                      658.97
#     Wheat   North     Sandy                      654.23
#    Cotton Central      Silt                      654.21
#     Wheat    West      Silt                      648.80
#     Wheat Central     Sandy                      645.86
#    Cotton   North      Clay                      645.61
#    Cotton Central      Clay                      636.08
#    Cotton    East      Clay                      614.38
#    Cotton   South     Loamy                      613.37
#     Wheat   South     Sandy                      611.82
#    Cotton   South      Silt                      610.85
#    Cotton Central     Sandy                      600.61
#    Cotton   South     Sandy                      585.46
#     Wheat    West      Clay                      568.81
#    Cotton    East     Loamy                      567.81
#    Cotton   South      Clay                      552.51
#    Cotton    East      Silt                      542.72
#    Cotton    West     Sandy                      536.80
#    Cotton    West     Loamy                      533.67
#    Cotton    West      Clay                      522.71
#    Cotton   North      Silt                      511.95
# """

# %% what region produces the most yields?
yield_region = (
    df.groupby(['Region'], as_index=False)
      ['Crop_Yield_ton_per_hectare']
      .sum()
      .sort_values(by='Crop_Yield_ton_per_hectare', ascending=False)
)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(yield_region.to_string(index=False))

# """ 
# Region  Crop_Yield_ton_per_hectare
#    west                    46107.60
#   south                    45427.43
#    east                    45004.99
# central                    44676.20
#   north                    42211.79
#   """
# %% count of dirt type per region? 
soil_pivot = pd.crosstab(
    df['Region'], #x-axis
    df['Soil_Type'] #y-axis what is being counted
)

soil_pivot
# """
# Soil_Type  clay  loamy  sandy  silt
# Region                             
# central     509    495    481   497
# east        471    515    527   493
# north       511    529    494   469
# south       482    500    510   521
# west        494    504    484   514
# """

# %% what dirt makes the most crops
soil_crops_pivot = pd.pivot_table(
    df,
    index='Crop_Type',               # rows
    columns='Soil_Type',          # columns
    values='Crop_Yield_ton_per_hectare',  # what to sum
    aggfunc='sum',                # sum the yields
    fill_value=0                  # replace NaN with 0
)

print(soil_crops_pivot.to_string())

# """
# Soil_Type      clay     loamy     sandy      silt
# Crop_Type                                        
# cotton      2971.29   3122.01   3274.18   3007.47
# maize       4138.32   3864.20   4532.68   4170.87
# potato     10121.99  10916.10  10533.49   9996.41
# rice        3864.06   4386.60   3944.28   3807.31
# sugarcane  31242.69  31474.68  27301.83  32178.61
# wheat       3630.06   3706.26   3482.95   3759.67
# """

# %% does more fertilizer=more yield? (not really)
var1= 'Fertilizer_Used'
var2= 'Crop_Yield_ton_per_hectare'

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

# %% yield above 60
above_60_yield= df[df['Crop_Yield_ton_per_hectare']>60]

above_60_yield.to_csv("above_60_yield.csv", index=False)

print("CSV saved as 'above_60_yield.csv'")

# %% rainfall_above_mean_less_fertilizer regression

#['Maize' 'Potato' 'Rice' 'Sugarcane' 'Wheat' 'Cotton']

# Crop selection
crop = 'Potato'
var1 = 'N'
var2 = 'Crop_Yield_ton_per_hectare'
Constant = 'Soil_Moisture'

# Relaxed filtering: Rainfall >= mean, Fertilizer <= 75th percentile
fert_thresh = df['Fertilizer_Used'].quantile(0.75)
rainfall_above_mean_less_fertilizer = df[
    (df['Rainfall'] >= df['Rainfall'].mean())&
    (df['Crop_Type'] == crop)
]

if len(rainfall_above_mean_less_fertilizer) < 2:
    print(f"Not enough data for {crop} with the current filtering criteria.")
else:
    # -------------------------
    # Regression
    # -------------------------
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # Split training and test datasets
    np.random.seed(42)
    rainfall_above_mean_less_fertilizer['r'] = np.random.uniform(size=len(rainfall_above_mean_less_fertilizer))
    train = rainfall_above_mean_less_fertilizer[rainfall_above_mean_less_fertilizer['r'] <= 0.6].drop(columns='r')
    test = rainfall_above_mean_less_fertilizer[rainfall_above_mean_less_fertilizer['r'] > 0.6].drop(columns='r')

    # Regression on training data
    y = train[var2]
    x = sm.add_constant(train[var1])  # add intercept
    reg = sm.OLS(y, x).fit()
    print(reg.summary())

    # Test predictions
    tx = sm.add_constant(test[var1])
    psales = reg.predict(tx)
    tsales = test[var2]
    ae = abs(tsales - psales)
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean((tsales - psales) ** 2))
    print(f"Mean absolute error (MAE): {mae}")
    print(f"Root mean square error (RMSE): {rmse}")

    # Prediction interval plot
    from statsmodels.stats.outliers_influence import summary_table
    st, data, ss2 = summary_table(reg, alpha=0.05)
    fittedvalues = data[:, 2]
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    fig, ax = plt.subplots()
    ax.plot(train[var1], y, 'o')
    ax.plot(train[var1], fittedvalues, '-', lw=2)
    ax.plot(train[var1], predict_ci_low, 'r--', lw=2)
    ax.plot(train[var1], predict_ci_upp, 'r--', lw=2)
    ax.set_title(f"Predicting {var2} from {var1} for {crop}")
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    plt.show()

    # -------------------------
    # Clustering
    # -------------------------
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score
    import seaborn as sns

    tocluster = rainfall_above_mean_less_fertilizer[[var1, var2]]

    # Evaluate Calinski–Harabasz index for 2-6 clusters
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(tocluster)
        ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
        print(f"Calinski–Harabasz Index for {i} clusters:", ch_score)

    # Use 5 clusters as example
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(tocluster)
    clusters = kmeans.predict(tocluster)
    rainfall_above_mean_less_fertilizer['cluster'] = clusters

    # Cluster summary
    p1 = rainfall_above_mean_less_fertilizer.pivot_table(index='cluster', values=[var1, var2], aggfunc='mean')
    p2 = rainfall_above_mean_less_fertilizer.pivot_table(index='cluster', values=[Constant], aggfunc='count')
    print("Cluster Means:\n", p1)
    print(f"\nNumber of {Constant} per cluster:\n", p2)

    # Plot clusters
    sns.set_style('white')
    customPalette = ['red', 'blue', 'green', 'yellow', 'black']
    sns.set_palette(customPalette)

    sns.lmplot(
        data=rainfall_above_mean_less_fertilizer,
        x=var1,
        y=var2,
        hue='cluster',
        fit_reg=False,
        legend=True
    )
    plt.show()

# %% does more pesticied=more yield?

#['Maize' 'Potato' 'Rice' 'Sugarcane' 'Wheat' 'Cotton']

# Crop selection
crop = 'Potato'
var1 = 'Pesticide_Used'
var2 = 'Crop_Yield_ton_per_hectare'
Constant = 'Soil_Moisture'

# Relaxed filtering: Rainfall >= mean, Fertilizer <= 75th percentile
fert_thresh = df['Fertilizer_Used'].quantile(0.75)
rainfall_above_mean_less_fertilizer = df[
    (df['Rainfall'] >= df['Rainfall'].mean())&
    (df['Crop_Type'] == crop)
]

if len(rainfall_above_mean_less_fertilizer) < 2:
    print(f"Not enough data for {crop} with the current filtering criteria.")
else:
    # -------------------------
    # Regression
    # -------------------------
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # Split training and test datasets
    np.random.seed(42)
    rainfall_above_mean_less_fertilizer['r'] = np.random.uniform(size=len(rainfall_above_mean_less_fertilizer))
    train = rainfall_above_mean_less_fertilizer[rainfall_above_mean_less_fertilizer['r'] <= 0.6].drop(columns='r')
    test = rainfall_above_mean_less_fertilizer[rainfall_above_mean_less_fertilizer['r'] > 0.6].drop(columns='r')

    # Regression on training data
    y = train[var2]
    x = sm.add_constant(train[var1])  # add intercept
    reg = sm.OLS(y, x).fit()
    print(reg.summary())

    # Test predictions
    tx = sm.add_constant(test[var1])
    psales = reg.predict(tx)
    tsales = test[var2]
    ae = abs(tsales - psales)
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean((tsales - psales) ** 2))
    print(f"Mean absolute error (MAE): {mae}")
    print(f"Root mean square error (RMSE): {rmse}")

    # Prediction interval plot
    from statsmodels.stats.outliers_influence import summary_table
    st, data, ss2 = summary_table(reg, alpha=0.05)
    fittedvalues = data[:, 2]
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    fig, ax = plt.subplots()
    ax.plot(train[var1], y, 'o')
    ax.plot(train[var1], fittedvalues, '-', lw=2)
    ax.plot(train[var1], predict_ci_low, 'r--', lw=2)
    ax.plot(train[var1], predict_ci_upp, 'r--', lw=2)
    ax.set_title(f"Predicting {var2} from {var1} for {crop}")
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    plt.show()

    # -------------------------
    # Clustering
    # -------------------------
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score
    import seaborn as sns

    tocluster = rainfall_above_mean_less_fertilizer[[var1, var2]]

    # Evaluate Calinski–Harabasz index for 2-6 clusters
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(tocluster)
        ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
        print(f"Calinski–Harabasz Index for {i} clusters:", ch_score)

    # Use 5 clusters as example
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(tocluster)
    clusters = kmeans.predict(tocluster)
    rainfall_above_mean_less_fertilizer['cluster'] = clusters

    # Cluster summary
    p1 = rainfall_above_mean_less_fertilizer.pivot_table(index='cluster', values=[var1, var2], aggfunc='mean')
    p2 = rainfall_above_mean_less_fertilizer.pivot_table(index='cluster', values=[Constant], aggfunc='count')
    print("Cluster Means:\n", p1)
    print(f"\nNumber of {Constant} per cluster:\n", p2)

    # Plot clusters
    sns.set_style('white')
    customPalette = ['red', 'blue', 'green', 'yellow', 'black']
    sns.set_palette(customPalette)

    sns.lmplot(
        data=rainfall_above_mean_less_fertilizer,
        x=var1,
        y=var2,
        hue='cluster',
        fit_reg=False,
        legend=True
    )
    plt.show()


# %% thoughts ideas on what i want to see
'''
Write up: 
    In the data I have found the more water and the more sunlight the crop gets the better it will grow and increase yields,  
    this is very obvious. Depending on the crop or on the soil type it does matter where it is grown. For example, sugarcane grows 
    better in hard ground (clay), while corn grows better on sandy ground. If I was to grow crops i would grow most of them anywhere 
    but up north. It shows a limitied number of crops to grow, and very little yeilds. 
'''

# %%

print(df['Crop_Type'].unique())