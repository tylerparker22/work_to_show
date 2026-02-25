# baseball analytics- Business Intelligence
# %% df
import pandas as pd
# For 2024 season data (most recent complete season)
df=pd.read_csv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Sports/statcast_2024.csv")

# %% df columns
print(df.columns.tolist())
# ['pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z', 'player_name', 
#  'batter', 'pitcher', 'events', 'description', 'spin_dir', 'spin_rate_deprecated', 
#  'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des', 'game_type', 'stand', 
#  'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 
#  'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 
#  'inning', 'inning_topbot', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 
#  'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 
#  'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 
#  'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 
#  'fielder_8', 'fielder_9', 'release_pos_y', 'estimated_ba_using_speedangle', 
#  'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 
#  'iso_value', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'pitch_name', 
#  'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score', 
#  'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 
#  'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed', 'swing_length', 
#  'estimated_slg_using_speedangle', 'delta_pitcher_run_exp', 'hyper_speed', 'home_score_diff', 
#  'bat_score_diff', 'home_win_exp', 'bat_win_exp', 'age_pit_legacy', 'age_bat_legacy', 'age_pit', 
#  'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat', 
#  'pitcher_days_since_prev_game', 'batter_days_since_prev_game', 
#  'pitcher_days_until_next_game', 'batter_days_until_next_game', 'api_break_z_with_gravity', 
#  'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'attack_angle', 'attack_direction', 
#  'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 
#  'intercept_ball_minus_batter_pos_y_inches']

# %% cluster example - Balls Put in Play Analysis
var1 = 'launch_speed'
var2 = 'launch_angle'
Constant = 'p_throws'
# filt_df shape
filt_df = df.dropna(subset=['launch_speed', 'launch_angle']).copy()
filt_df.shape # (95148, 118)

#k-means
from sklearn.cluster import KMeans
tocluster=filt_df[[var1,var2]]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(tocluster)

#Apply CH to Different Numbers of Clusters:
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
tocluster=filt_df[[var1,var2]]
for i in range(2,7):
  kmeans = KMeans(n_clusters=i,random_state=42)
  kmeans.fit(tocluster)
  ch_score = calinski_harabasz_score(tocluster, kmeans.labels_)
  print("Calinski–Harabasz Index for",i,"clusters:", ch_score)
  
#Continuing with 5 clusters
from sklearn.cluster import KMeans
tocluster=filt_df[[var1,var2]]
kmeans = KMeans(n_clusters=5,random_state=42)
kmeans.fit(tocluster)
clusters = kmeans.predict(tocluster)

#Look at cluster membership
filt_df['cluster'] = clusters.tolist()
print(filt_df)

#cluster summary
p1=filt_df.pivot_table(index="cluster",values=[var1,var2],aggfunc="mean")
p2=filt_df.pivot_table(index="cluster",values=[Constant],aggfunc="count")
print("Cluster Means \n",p1)
print("\n \n Number of, {Constant}, per Cluster \n",p2)

#plotting clusters
import seaborn as sns
import matplotlib.pyplot as plt
filt_df["Clusters"]=clusters
#set font size of labels on matplotlib plots
plt.rc('font', size=8)
sns.set_style('white')
#set the colors used in the graph, see https://xkcd.com/color/rgb/
customPalette = ['red', 'blue', 'green', 'yellow', 'black' ]
sns.set_palette(customPalette)
#plot data with seaborn
facet = sns.lmplot(data=filt_df, x=var1, y=var2, hue='Clusters',
fit_reg=False, legend=True)
# %% Time Series Example 
hitter = 592450
description = 'hit_into_play'
#power = ['field_out', 'single', 'double', 'triple', 'home_run']   # must be a list

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# convert date column
df['game_date'] = pd.to_datetime(df['game_date'])

# date range
start_date = '2024-01-01'
end_date   = '2024-12-31'

# filter dataframe
ts_df = df[
    (df['batter'] == hitter) &
    (df['description'] == description) &
    #(df['events'].isin(power)) &
    (df['game_date'].between(start_date, end_date))
].copy()

# sort by time (VERY IMPORTANT)
ts_df = ts_df.sort_values('game_date')

# plot
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(ts_df['game_date'], ts_df['launch_angle'],
        color="black", linestyle="solid", label="Launch Angle");

ax.plot(ts_df['game_date'], ts_df['launch_speed'],
        color="red", linestyle="solid", label="Launch Speed");

ax.legend()

# rotate 45 degrees
plt.xticks(rotation=45)

plt.show()

# %% Random Forest Example
# Check which columns have data
print("Checking data availability:")
print(f"Total rows in df: {len(df)}")
print(f"Non-null break_angle_deprecated: {df['break_angle_deprecated'].notna().sum()}")
print(f"Non-null release_spin_rate: {df['release_spin_rate'].notna().sum()}")
print(f"Non-null launch_angle: {df['launch_angle'].notna().sum()}")

# Use a non-deprecated column as target
VALUES = ["pitch_type", "release_speed", "release_pos_x", "release_pos_z"]
TARGET_COL = "release_spin_rate"  # Use this instead of deprecated column

# Drop rows with NaN in relevant columns
df_rf = df.dropna(subset=VALUES + [TARGET_COL]).copy()

print(f"\nRows after dropping NaN: {len(df_rf)}")

if len(df_rf) == 0:
    print("ERROR: No data remaining after dropping NaN. Try different columns.")
else:
    # Divide data into train and test sets
    import numpy as np
    np.random.seed(42)
    df_rf['r'] = np.random.uniform(size=len(df_rf))
    train = df_rf[df_rf["r"] <= .6]
    test = df_rf[df_rf["r"] > .6]
    
    # Random Forest Regression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    # Encode pitch_type
    le = LabelEncoder()
    train_encoded = train.copy()
    test_encoded = test.copy()
    train_encoded['pitch_type'] = le.fit_transform(train['pitch_type'])
    test_encoded['pitch_type'] = le.transform(test['pitch_type'])
    
    X = train_encoded[VALUES]
    y = train_encoded[TARGET_COL]
    Xtest = test_encoded[VALUES]
    ytest = test_encoded[TARGET_COL]
    
    # Fit model
    rf = RandomForestRegressor(n_estimators=500, random_state=17)
    rf.fit(X, y)
    
    # Predict
    predictions = rf.predict(Xtest)
    
    # Evaluate
    print(f"\nR² Score: {r2_score(ytest, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(ytest, predictions)):.4f}")
    
    # Feature Importance
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = X.columns
    
    print("\nFeature Importance:")
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[sorted_indices[f]], 
                                importances[sorted_indices[f]]))
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), feat_labels[sorted_indices], rotation=45)
    plt.tight_layout()
    plt.show()
# -----------------------------------------
# %% regression

#split to training and test datasets
import numpy as np
np.random.seed(42)
df['r'] = np.random.uniform(size=len(df))
train = df[df["r"] <= .6]
test = df[df["r"] > .6]
train = train.drop("r", axis=1)
test = test.drop("r", axis=1)

# NOW clean the data (after splitting)
train[var1] = pd.to_numeric(train[var1], errors='coerce')
train[var2] = pd.to_numeric(train[var2], errors='coerce')
test[var1] = pd.to_numeric(test[var1], errors='coerce')
test[var2] = pd.to_numeric(test[var2], errors='coerce')

# Remove rows with NaN values created by conversion
train = train.dropna(subset=[var1, var2])
test = test.dropna(subset=[var1, var2])

#regression on training data
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Convert to standard numpy float64 (this is the key fix!)
y = train[var2].astype('float64')
x = train[var1].astype('float64')

x = sm.add_constant(x) #needed to force an intercept into the model
reg = sm.OLS(y, x).fit()
print(reg.summary())

#mean of absolute error (MAE) for test data
tx = test[var1].astype('float64')
tx = sm.add_constant(tx)
psales = reg.predict(tx)
tsales = test[var2].astype('float64')
ae = abs(tsales - psales)
mae = np.mean(ae)
print("The mean absolute error is", mae)

#Root Mean Square Error of test predictions
N = len(test)
rmse = np.sqrt((np.sum((tsales - psales)**2)) / N)
print("The root mean square error is", rmse)

#Prediction interval plot
#-------------------
#remove the intercept for the scatterplot
x = train[var1].astype('float64')
#import another regression module
from statsmodels.stats.outliers_influence import summary_table
#create a table of regression information from our regression problem
st, data, ss2 = summary_table(reg, alpha=0.05)
#extract the predicted values to create the regression line on the graph
fittedvalues = data[:, 2]
#extract the confidence intervals for an individual value (prediction intervals)
predict_ci_low, predict_ci_upp = data[:, 6:8].T
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
temp = ax.plot(x, y, 'o')
temp = ax.plot(x, fittedvalues, '-', lw=2)
temp = ax.plot(x, predict_ci_low, 'r--', lw=2)
temp = ax.plot(x, predict_ci_upp, 'r--', lw=2)
temp = ax.set_title(f"Predicting {var2} from {var1}")
temp = ax.set_xlabel(var1)
temp = ax.set_ylabel(var2)

# %% defensive positioning
# who shades the most? 
df['if_fielding_alignment']
df['of_fielding_alignment']

