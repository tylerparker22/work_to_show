# what i am doing...
# %% imports
import sqlite3
import pandas as pd
# %% CREATE dataframes 
DB_PATH = "C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Arkansas Tech Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
not_bus_page = pd.read_sql_query("SELECT * FROM post WHERE `Account username` <> 'atu_business'", conn) 
equal_bus_page = pd.read_sql_query("SELECT * FROM post WHERE `Account username` = 'atu_business'", conn)
all_data=pd.read_sql_query("SELECT * FROM post", conn)
conn.close()
not_bus_page.head()

# split 'Publish time'
# not equal
not_bus_page['Publish time']=pd.to_datetime(not_bus_page['Publish time']) # make date-time 
not_bus_page['Date_Publish']=not_bus_page['Publish time'].dt.date # make date column 
not_bus_page['Time_Publish']=not_bus_page['Publish time'].dt.time # make time column 
not_bus_page['WD_Publish']=not_bus_page['Publish time'].dt.day_name() # make time column 

# equal
equal_bus_page['Publish time']=pd.to_datetime(equal_bus_page['Publish time']) # make date-time 
equal_bus_page['Date_Publish']=equal_bus_page['Publish time'].dt.date # make date column 
equal_bus_page['Time_Publish']=equal_bus_page['Publish time'].dt.time # make time column 
equal_bus_page['WD_Publish']=equal_bus_page['Publish time'].dt.day_name() # make time column

# all data
all_data['Publish time']=pd.to_datetime(all_data['Publish time']) # make date-time 
all_data['Date_Publish']=all_data['Publish time'].dt.date # make date column 
all_data['Time_Publish']=all_data['Publish time'].dt.time # make time column 
all_data['WD_Publish']=all_data['Publish time'].dt.day_name() # make time column

# text analysis 
from textblob import TextBlob
# not equal
not_bus_page["Description"] = not_bus_page["Description"].fillna("") # make sure column is clean
not_bus_page["Description"] = not_bus_page["Description"].astype(str) # make sure string
not_bus_page["sentiment"] = not_bus_page["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
not_bus_page["subjectivity"] = not_bus_page["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
not_bus_page["len_chars"] = not_bus_page["Description"].apply(len)
not_bus_page["len_words"] = not_bus_page["Description"].apply(lambda x: len(x.split()))

# equal
equal_bus_page["Description"] = equal_bus_page["Description"].fillna("") # make sure column is clean
equal_bus_page["Description"] = equal_bus_page["Description"].astype(str) # make sure string
equal_bus_page["sentiment"] = equal_bus_page["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
equal_bus_page["subjectivity"] = equal_bus_page["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
equal_bus_page["len_chars"] = equal_bus_page["Description"].apply(len)
equal_bus_page["len_words"] = equal_bus_page["Description"].apply(lambda x: len(x.split()))

# all data
all_data["Description"] = all_data["Description"].fillna("") # make sure column is clean
all_data["Description"] = all_data["Description"].astype(str) # make sure string
all_data["sentiment"] = all_data["Description"].apply(lambda x: TextBlob(x).sentiment.polarity)
all_data["subjectivity"] = all_data["Description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
all_data["len_chars"] = all_data["Description"].apply(len)
all_data["len_words"] = all_data["Description"].apply(lambda x: len(x.split()))

# %% averge var 
# not equal to Bus Page
var= 'Views' # 'Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows'

not_bus_page[var]=not_bus_page[var].astype(float) # make sure it is numeric

var_analyze_not_equal=not_bus_page[var].mean()
print(round(var_analyze_not_equal,2)) # round 2 dec places

# equal to bus page
equal_bus_page[var]=equal_bus_page[var].astype(float) # make sure it is numeric

var_analyze_equal=equal_bus_page[var].mean()
print(round(var_analyze_equal,2)) # round 2 dec places

# all data
all_data[var]=all_data[var].astype(float) # make sure it is numeric

var_analyze_all=all_data[var].mean()
print(round(var_analyze_all,2)) # round 2 dec places
# %% count
# not equal to Bus Page
var= 'Views' # 'Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows'

not_bus_page[var]=not_bus_page[var].astype(float) # make sure it is numeric

var_analyze_not_equal=not_bus_page[var].count()
print(round(var_analyze_not_equal,2)) # round 2 dec places

# equal to bus page
equal_bus_page[var]=equal_bus_page[var].astype(float) # make sure it is numeric

var_analyze_equal=equal_bus_page[var].count()
print(round(var_analyze_equal,2)) # round 2 dec places

# all data
all_data[var]=all_data[var].astype(float) # make sure it is numeric

var_analyze_all=all_data[var].count()
print(round(var_analyze_all,2)) # round 2 dec places

# %% column names
print(all_data.columns.tolist())
# ['Post ID', 'Account ID', 'Account username', 'Account name', 'Description', 
#  'Duration (sec)', 'Publish time', 'Permalink', 'Post type', 'Data comment', 
#  'Date', 'Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows', 
#  'Date_Publish', 'Time_Publish', 'WD_Publish', 'sentiment', 'subjectivity', 
#  'len_chars', 'len_words']
# %% (2 way) Anova
from scipy import stats
# =============================================================================
# specific stuff 
column_to_find= 'Post type' # specific column 
var_in_col1= 'IG reel' # specific in column 
var_in_col2= 'IG carousel' # specific in column 
var_in_col3= 'IG image' # specific in column 
# =============================================================================
# =============================================================================
# variable to find
var_to_find = 'Likes' #  'Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows'
# =============================================================================
# not business page
not_bus_page[var_to_find]=not_bus_page[var_to_find].astype(float) # convert to numeric

group1_not = not_bus_page[not_bus_page[column_to_find] == var_in_col1][var_to_find].dropna()
group2_not = not_bus_page[not_bus_page[column_to_find] == var_in_col2][var_to_find].dropna()
group3_not = not_bus_page[not_bus_page[column_to_find] == var_in_col3][var_to_find].dropna()

# run ANOVA
f_stat, p_val = stats.f_oneway(group1_not, group2_not, group3_not)

print("F-stat:", f_stat)
print("p-value:", p_val)

# -----------------------------------------------------------------------------
# is business page
# convert to numeric
equal_bus_page[var_to_find]=equal_bus_page[var_to_find].astype(float) # convert to numeric

group1_equal = equal_bus_page[equal_bus_page[column_to_find] == var_in_col1][var_to_find].dropna()
group2_equal = equal_bus_page[equal_bus_page[column_to_find] == var_in_col2][var_to_find].dropna()
group3_equal = equal_bus_page[equal_bus_page[column_to_find] == var_in_col3][var_to_find].dropna()

# run ANOVA
f_stat, p_val = stats.f_oneway(group1_equal, group2_equal, group3_equal)

print("F-stat:", f_stat)
print("p-value:", p_val)
# -----------------------------------------------------------------------------
# convert to numeric
all_data[var_to_find]=all_data[var_to_find].astype(float) # convert to numeric

group1_all = all_data[all_data[column_to_find] == var_in_col1][var_to_find].dropna()
group2_all = all_data[all_data[column_to_find] == var_in_col2][var_to_find].dropna()
group3_all = all_data[all_data[column_to_find] == var_in_col3][var_to_find].dropna()

# run ANOVA
f_stat, p_val = stats.f_oneway(group1_all , group2_all , group3_all )

print("F-stat:", f_stat)
print("p-value:", p_val)

# check desc stats to see particular diff (Compare anova)
print("not equal")
print("Reel Mean:", group1_not.mean(), ",Reel Median:", group1_not.median(), ", Number in Group:", len(group1_not))
print("Carousel Mean:", group2_not.mean(), ", Carousel Median:", group2_not.median(), ", Number in Group:", len(group2_not))
print("Image Mean:", group3_not.mean(), ", Image Median:", group3_not.median(), ", Number in Group:", len(group3_not))
# -----------------------------------------------------------------------------

print('equal')
print("Reel Mean:", group1_equal.mean(), ",Reel Median:", group1_equal.median(), ", Number in Group:", len(group1_equal))
print("Carousel Mean:", group2_equal.mean(), ", Carousel Median:", group2_equal.median(), ", Number in Group:", len(group2_equal))
print("Image Mean:", group3_equal.mean(), ", Image Median:", group3_equal.median(), ", Number in Group:", len(group3_equal))
# -----------------------------------------------------------------------------

print('all')
print("Reel Mean:", group1_all.mean(), ",Reel Median:", group1_all.median(), ", Number in Group:", len(group1_all))
print("Carousel Mean:", group2_all.mean(), ", Carousel Median:", group2_all.median(), ", Number in Group:", len(group2_all))
print("Image Mean:", group3_all.mean(), ", Image Median:", group3_all.median(), ", Number in Group:", len(group3_all))

# tukey test (For anova comparison)
# imports
from scipy. stats import tukey_hsd
res_not= tukey_hsd(group1_not, group2_not, group3_not)
print(res_not)

res_equal= tukey_hsd(group1_equal, group2_equal, group3_equal)
print(res_equal)

res_all= tukey_hsd(group1_all, group2_all, group3_all)
print(res_all)