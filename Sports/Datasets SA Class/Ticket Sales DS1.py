# %% data set 1 SAC
import pandas as pd

df=pd.read_csv("D:/Rstudio/Sports Analytics/DS1 SA Class/SA_DS1.csv")


# %% column names
print(df.columns)
"""
['Opponent', 'SeasonStart', 'SeasonEnd', 'EventDate', 'WeekDay',
       'EventTime', 'SoldDaysBeforeEvent', 'Level', 'Position', 'SectionName',
       'CustomerUniqueID', 'MethodofDeliveryType', 'Market', 'TicketsSold',
       'AverageTicketPrice', 'TotalSale']
"""

numeric_columns=['SoldDaysBeforeEvent', 	"TicketsSold",	'AverageTicketPrice',	'TotalSale']
for col in numeric_columns:
    # Convert the column to numeric (int or float)
    df[col] = pd.to_numeric(df[col], errors='coerce')
# %% thoughts
"""
descriptive stats
what is the best day to sale tickets?
what is the most desired position? 
"""
# %% avg 'Opponent','Level', 'SectionName'

avg_o_l_sn = (
    df
    .groupby(['Opponent','Level', 'SectionName'])[
        ['SoldDaysBeforeEvent', 'TicketsSold', 'TotalSale']
    ]
    .mean()
)

print(avg_o_l_sn)

# %% what days do tickets sale the most? 

#turn 'EventDate' to date column 
df['EventDate'] = pd.to_datetime(df['EventDate'])

#what is the day of the week 
df['DayOfWeek'] = df['EventDate'].dt.day_name()

best_sell_day=(
    df.groupby(['SeasonStart','DayOfWeek']).agg({'TicketsSold':'sum'})
    .sort_values(by='TicketsSold', ascending=False))
best_sell_day

"""
                       TicketsSold
SeasonStart DayOfWeek             
2018        Saturday         18780
            Thursday         11324
2017        Saturday          8846
            Thursday          6722
2016        Saturday          4627
            Thursday          4378
2018        Monday            4287
2017        Sunday            4165
2018        Tuesday           3193
2017        Monday            3072
            Tuesday           2763
2018        Sunday            2625
2017        Wednesday         1532
2016        Wednesday         1456
            Monday            1302
            Tuesday           1265
            Friday            1119
2018        Friday             964
            Wednesday          832
2016        Sunday             695
"""

# %% what is the most desired position? 

desired_pos=(df.groupby(['SeasonStart','Level','Position'])
             .agg({'TicketsSold':'count'})
             .sort_values(by='TicketsSold', ascending=False))
desired_pos

"""
                             TicketsSold
SeasonStart Level  Position             
2018        UPPER  CORNER           4338
2017        UPPER  CORNER           3609
2018        UPPER  CENTER           2287
            LOWER  CENTER           2201
            MIDDLE CENTER           2197
            LOWER  CORNER           2164
2016        UPPER  CORNER           2057
2017        MIDDLE CENTER           1827
            UPPER  CENTER           1795
            LOWER  CORNER           1680
2018        LOWER  END              1518
2016        LOWER  CORNER            899
            MIDDLE CENTER            867
            UPPER  CENTER            809
2017        LOWER  CENTER            767
            MIDDLE CORNER            725
            LOWER  END               708
2016        LOWER  CENTER            638
2018        UPPER  END               582
2016        MIDDLE CORNER            548
            LOWER  END               385
2017        UPPER  END               307
2018        MIDDLE CORNER            243
2016        UPPER  END               116
"""
# %% why is the level=upper and positon=corner so much more ts? 
#best sales upper corner 
upper_corner_price = round(df.loc[
    (df['Level'] == 'UPPER') &
    (df['Position'] == 'CORNER'),
    'SectionName'
].count(),2)

upper_corner_price
"""
49.41
"""

#compare to upper end least sales
upper_end_price = round(df.loc[
    (df['Level'] == 'UPPER') &
    (df['Position'] == 'END'),
    'AverageTicketPrice'
].mean(),2)

upper_end_price
"""
42.5
"""

# %% what team will i make this most money off of 
max_rev = (
    df.groupby(['Opponent', 'SeasonStart'], as_index=False)
      .agg({
          'TotalSale': 'sum',
          'AverageTicketPrice': 'mean'   
      })
      .sort_values(by='TotalSale', ascending=False)
      .round(2)
)

with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
    'display.max_colwidth', None
):
    print(max_rev.to_string(index=False))

# %% create a profit column 

profit_df = (
    df.groupby(['Opponent', 'SeasonStart'], as_index=False)
      .agg(
          TotalSale=('TotalSale', 'sum'),
          AvgTicketPrice=('AverageTicketPrice', 'mean')
      )
)

profit_df['profit'] = profit_df['TotalSale'] * profit_df['AvgTicketPrice']
                                     
df.head()

# %% graph profit column

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Fill NaN with 0 in numeric columns
df[numeric_columns] = df[numeric_columns].fillna(0)

# Loop through all numeric columns and plot them individually as bar charts
for col in numeric_columns:
    plt.figure(figsize=(12,5))
    
    # Plot numeric column as a bar chart
    plt.bar(df['Opponent'], df[col], color='red')
    
    plt.title(f"{col} over Opponent")
    plt.xlabel("Opponent")
    plt.ylabel(col)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plain numbers for y-axis
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().ticklabel_format(style='plain', axis='y')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

"""
4,12 have the lowest tickets sold under 100 days run a promotion to get those tickets gone
17,12,29,11,6 are all <1500 ts
2 must be a rival has most cost 
2,13,30,31 have the most avg ticket price
sold the most tickets to 8, 26, 31
"""
# %% tickets sold per opponent 
ts_opp=(
        df.groupby('Opponent')
        .agg({'TicketsSold':'sum'})
        .sort_values(by='TicketsSold', ascending=True)
        )
ts_opp
"""
<1500 ts
          TicketsSold
Opponent             
17               1091
12               1302
29               1380
11               1393
6                1472
"""
# %% weekday profit

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Fill NaN with 0 in numeric columns
df[numeric_columns] = df[numeric_columns].fillna(0)

# Loop through all numeric columns and plot them individually as bar charts
for col in numeric_columns:
    plt.figure(figsize=(12,5))
    
    # Plot numeric column as a bar chart
    plt.bar(df['DayOfWeek'], df[col], color='blue')
    
    plt.title(f"{col} per Week Day")
    plt.xlabel("DayOfWeek")
    plt.ylabel(col)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plain numbers for y-axis
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().ticklabel_format(style='plain', axis='y')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
"""
try and run promo on fridays to get more tickets sold that day, 
thursday cost is the most try to incentivize that day
saturday is the best day, make more profit increase cost on sat have the most traffic
"""

# %% when ts <1500
df_2016 = df[df['SeasonStart'] == 2016]
df_2017 = df[df['SeasonStart'] == 2017]
df_2018 = df[df['SeasonStart'] == 2018]

# Sum tickets sold per WeekDay and EventTime
tickets_by_time_2016=(
    df_2016
    .groupby(['Opponent', 'WeekDay', 'EventTime'], as_index=False)['TicketsSold']
    .sum()
    .sort_values(by='TicketsSold', ascending=False)
)

tickets_by_time_2017=(
    df_2017
    .groupby(['Opponent', 'WeekDay', 'EventTime'], as_index=False)['TicketsSold']
    .sum()
    .sort_values(by='TicketsSold', ascending=False)
)

tickets_by_time_2018=(
    df_2018
    .groupby(['Opponent', 'WeekDay', 'EventTime'], as_index=False)['TicketsSold']
    .sum()
    .sort_values(by='TicketsSold', ascending=False)
)

def agg_by_weekday(df):
    return (
        df.groupby('WeekDay', as_index=False)['TicketsSold']
          .sum()
          .sort_values('WeekDay')
    )

d16 = agg_by_weekday(df_2016)
d17 = agg_by_weekday(df_2017)
d18 = agg_by_weekday(df_2018)

weekdays = d16['WeekDay']
x = range(len(weekdays))

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.figure(figsize=(12,6))

plt.bar(x, d16['TicketsSold'], 
        color='navy', alpha=0.9, label='2016')

plt.bar(x, d17['TicketsSold'], 
        color='blue', alpha=0.6, label='2017')

plt.bar(x, d18['TicketsSold'], 
        color='skyblue', alpha=0.4, label='2018')

plt.xticks(x, weekdays, rotation=45, ha='right')
plt.xlabel("WeekDay")
plt.ylabel("Tickets Sold")
plt.title("Tickets Sold by WeekDay (Overlayed by Year)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.tight_layout()
plt.show()
