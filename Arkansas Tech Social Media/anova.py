# %% imports
import sqlite3
import pandas as pd
# %% connect to db
DB_PATH = "C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Arkansas Tech Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post", conn) 
conn.close()
print(df.columns.tolist())
# ['Post ID', 'Account ID', 
# 'Account username', 'Account name', 'Description', 
# 'Duration (sec)', 'Publish time', 'Permalink', 'Post type', 
# 'Data comment', 'Date', 'Views', 'Likes', 'Shares', 'Comments', 
# 'Saves', 'Reach', 'Follows']
# -----------------------------------------------------------------------------
# %% (2 way) Anova
from scipy import stats

var_to_find = 'Reach' # Views Reach Follows
# convert to numeric
df[var_to_find]=df[var_to_find].astype(float)

group1 = df[df['Post type'] == "IG reel"][var_to_find].dropna()
group2 = df[df['Post type'] == "IG carousel"][var_to_find].dropna()
group3 = df[df['Post type'] == "IG image"][var_to_find].dropna()

# run ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)

print("F-stat:", f_stat)
print("p-value:", p_val)
# %% check desc stats to see particular diff (Compare anova)
print("Reel Mean:", group1.mean(), ",Reel Median:", group1.median(), ", Number in Group:", len(group1))
print("Carousel Mean:", group2.mean(), ", Carousel Median:", group2.median(), ", Number in Group:", len(group2))
print("Image Mean:", group3.mean(), ", Image Median:", group3.median(), ", Number in Group:", len(group3))

# F-stat: 4.281238681544712
# p-value: 0.01536430444542936
# Reel Mean: 2533.63 ,Reel Median: 1626.0 , Number in Group: 100
# Carousel Mean: 1601.578947368421 , Carousel Median: 1086.0 , Number in Group: 57
# Image Mean: 923.9285714285714 , Image Median: 839.5 , Number in Group: 14

# %% tukey test (For anova comparison)
# imports
from scipy. stats import tukey_hsd
res= tukey_hsd(group1, group2, group3)
print(res)

# Comparison  Statistic  p-value  Lower CI  Upper CI
#  (0 - 1)    932.051     0.063   -37.998  1902.100
#  (0 - 2)   1609.701     0.061   -58.198  3277.600
#  (1 - 0)   -932.051     0.063 -1902.100    37.998
#  (1 - 2)    677.650     0.629 -1065.798  2421.099
#  (2 - 0)  -1609.701     0.061 -3277.600    58.198
#  (2 - 1)   -677.650     0.629 -2421.099  1065.798

# Reels show ~932 more reach than carousels (borderline: p = 0.063)
# Reels show ~1,610 more reach than images (borderline: p = 0.061)
# -----------------------------------------------------------------------------