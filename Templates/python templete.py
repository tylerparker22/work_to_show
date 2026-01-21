# %% import a dataset
import pandas as pd
games = pd.read_csv('games.csv')
scores = pd.read_csv('scores.csv')
locations = pd.read_csv('locations.csv')

# %% head of the data
games.head()
# %% column names of the data
print(df.columns)

# %% merges

#inner
merged_df = pd.merge(left=games,right=scores, left_on='scores_id',right_on='id',how='inner')

#left 
merged_df = pd.merge(left=games,right=scores, left_on='scores_id',right_on='id',how='left')

#right
merged_df = pd.merge(left=games,right=scores, left_on='scores_id',right_on='id',how='right')

#outer
merged_df = pd.merge(left=games,right=scores, left_on='scores_id',right_on='id',how='outer')

# show output
merged_df.head()

#------------------------------------------------------------------------------
"""Perform a left merge with `shootouts` on the left and `goals` on the right. 
Use the date, home, and away team columns to correctly match shootouts with the 
in-game goals data in `goals`.
"""

merged_df = pd.merge(left=shootouts,
                     right=goals, 
                     left_on = ['shootout_date', 'home_team', 'away_team'], #matches columns of the left 
                     right_on = ['game_date', 'home_team', 'away_team'], #matches columns of the right
                     how='left') #type of merge
# %% filter for na
merged_df[merged_df['home_score'].isna()]

# %% group by a column with a function
name=df.groupby('column_to_groupby').agg({'column_to_manipulate':'function'})
largest_margin = results.groupby('home_team').agg({'win_margin':'mean'})

#make a groupby into a df
crop_region = (
    df.groupby(['Crop_Type', 'Region'], as_index=False)['Crop_Yield_ton_per_hectare']
      .sum()
      .sort_values(by='Crop_Yield_ton_per_hectare', ascending=False)
)

# %% rename a column
1. renaming `away_team` to `nunique_away_teams`

teams_per_country = results.groupby('country').agg(
    {'away_team': 'nunique'}
)
# rename column
teams_per_country.columns = ['nunique_away_teams']
# %% reseting the index=
teams_per_country = teams_per_country.reset_index()
# %% function for large group of data
miguel_cabrera_average = miguel_cabrera.mean(numeric_only=True)

# %% create new column in df
df['column_name']=

# %% averages for all columns quick 

# Dictionary to store averages
players_averages = {}

# Loop through each player's DataFrame
for name, df in zip(
        #all uniqe names in the data
    ['Miguel Cabrera', 'Albert Pujols', 'Alex Rodriguez', 'David Ortiz'],
    players_dfs
):
    # Compute mean of numeric columns (ignores NaN automatically)
    avg = df.mean(numeric_only=True).round(3)
    
    # Store in dictionary
    players_averages[name] = avg

# View results
for player, avg in players_averages.items():
    print(f"\n{player} averages:\n", avg)
    
# %% if statements

column='Total_Riders'

if rides[column].dtype == 'object':
    rides[column] = rides[column].str.lower()
elif rides[column].dtype == 'int64':
    rides[column] = rides[column] / 1000

rides.head()


3. We've defined a variable `total_riders` in the cell below. Run that cell first. In the following cell, write an `if/elif/else` statement:
- if `total_riders` is below `20000`, define a variable `pop = 'unpopular'`
- elif `total_riders` is below `100000`, define a variable `pop = 'popular'`
- otherwise, define a variable `pop = 'superpopular'`

Make sure to run your solution and then save the notebook before selecting `Test Work`!
total_riders = 100

## YOUR SOLUTION HERE ##
if total_riders < 20000:
    pop = 'unpopular'
elif total_riders < 100000:
    pop = 'popular'
else:
    pop = 'superpopular'

# show output
pop

# %% convert columns to numeric

for col in numeric_columns:
    # Convert the column to numeric (int or float)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# %% piviot table
soil_pivot = pd.crosstab(
    df['Region'], #x-axis
    df['Soil_Type'] #y-axis what is being counted
)

soil_pivot = pd.pivot_table(
    df,
    index='Region',               # rows
    columns='Soil_Type',          # columns
    values='Crop_Yield_ton_per_hectare',  # what to sum
    aggfunc='sum',                # sum the yields
    fill_value=0                  # replace NaN with 0
)

print(soil_pivot.to_string())
# %% get as csv

crop_region_sorted.to_csv("crop_region_sorted.csv", index=False)

# %% see all output
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(yield_region.to_string(index=False))

# %% create a dataframe 
opponent_season_totals = (
    df
    .groupby(['Opponent', 'SeasonStart', 'SeasonEnd'], as_index=False)
    .agg({'TicketsSold': 'sum'})
)

opponent_best_yr = (
    opponent_season_totals
    .sort_values('TicketsSold', ascending=False)
    .groupby('Opponent')
    .head(1)
    .reset_index(drop=True)
)
