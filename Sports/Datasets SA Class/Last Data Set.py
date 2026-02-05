# 3.13.7 python works
# %% import data
import pandas as pd
df=pd.read_csv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Sports/Datasets SA Class/SBR001-715.csv")
#----------------------------
# %% clean data
# Now row 0 should be the header row
df.columns = df.iloc[2]  # Set first row as header
df = df[1:]              # Remove that row from data
df = df.reset_index(drop=True)  # Reset index
#----------------------------
# Optional: remove extra spaces from column names
df.columns = df.columns.str.strip()
df.columns=df.columns.str.replace(',', '', regex=False)
#----------------------------
# reshape data 
df_melted = df.melt(id_vars=['Item'], var_name='Year', value_name='Value')
print(df_melted)
#----------------------------
# %% column names
df=df_melted
print(df.columns.tolist())
# ['Item', 'Year', 'Value']
#----------------------------
# %% summary stats
sum_stat=df.groupby('Item').agg({'Year':'unique'})
sum_stat.describe()
#----------------------------------
# %% Feature Importance for Each Item
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

TARGET_COL = "Value"

# Set random seed for reproducibility
np.random.seed(42)

# Create a clean copy
df_clean = df.copy()

# Remove commas from Value column and convert to numeric
if df_clean[TARGET_COL].dtype == 'object':
    df_clean[TARGET_COL] = df_clean[TARGET_COL].str.replace(',', '', regex=False)
    df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors='coerce')

# Remove commas from Year column and convert to numeric
if df_clean['Year'].dtype == 'object':
    df_clean['Year'] = df_clean['Year'].str.replace(',', '', regex=False)
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')

print(f"Original dataset size: {len(df)}")

# Remove rows where Target column is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"Dataset size after removing NaN in '{TARGET_COL}': {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} rows with NaN values/n")

# Get unique items
unique_items = df_clean['Item'].unique()
print(f"Number of unique items: {len(unique_items)}/n")

# Store results
results = []

# Loop through each unique item
for item in unique_items:
    print(f"/n{'='*60}")
    print(f"Processing Item: {item}")
    print(f"{'='*60}")
    
    # Filter data for this item
    item_data = df_clean[df_clean['Item'] == item].copy()
    
    # Check if we have enough data
    if len(item_data) < 10:
        print(f"Skipping {item} - insufficient data (only {len(item_data)} rows)")
        continue
    
    # Divide data into train and test sets
    item_data['r'] = np.random.uniform(size=len(item_data))
    train = item_data[item_data["r"] <= 0.6]
    test = item_data[item_data["r"] > 0.6]
    
    if len(train) < 5 or len(test) < 2:
        print(f"Skipping {item} - insufficient train/test split")
        continue
    
    print(f"Train set size: {len(train)}")
    print(f"Test set size: {len(test)}")
    
    # Prepare features and target (only Year as feature now)
    X = train[['Year']]
    y = train[TARGET_COL]
    Xtest = test[['Year']]
    ytest = test[TARGET_COL]
    
    # Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=17,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Evaluate model
    train_score = rf.score(X, y)
    test_score = rf.score(Xtest, ytest)
    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Store results
    results.append({
        'Item': item,
        'Train_R2': train_score,
        'Test_R2': test_score,
        'Train_Size': len(train),
        'Test_Size': len(test),
        'Feature_Importance_Year': rf.feature_importances_[0]
    })

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_R2', ascending=False)

print("/n" + "="*60)
print("SUMMARY RESULTS - Sorted by Test R²")
print("="*60)
print(results_df.to_string(index=False))

# Visualize results
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: R² Scores by Item
ax1 = axes[0]
x_pos = np.arange(len(results_df))
width = 0.35

ax1.bar(x_pos - width/2, results_df['Train_R2'], width, label='Train R²', alpha=0.8)
ax1.bar(x_pos + width/2, results_df['Test_R2'], width, label='Test R²', alpha=0.8)
ax1.set_xlabel('Item')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance by Item')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Item'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Feature Importance (Year) by Item
ax2 = axes[1]
ax2.bar(x_pos, results_df['Feature_Importance_Year'], color='steelblue', alpha=0.8)
ax2.set_xlabel('Item')
ax2.set_ylabel('Feature Importance')
ax2.set_title('Year Feature Importance by Item')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['Item'], rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Save results to CSV
results_df.to_csv('feature_importance_by_item.csv', index=False)
print("/nResults saved to 'feature_importance_by_item.csv'")



# %% what i am doing
#"""Based on MiLB fan engagement data from 2016–2024, which fan market segments should Minor League Baseball target 
#to maximize the success of the MiLB Community Impact Series (Baseball Futures 2026), and why?"""
df=df_clean

# %% target graphed over time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# Define the items you want to plot (these are row names in the 'Item' column)
data1 = 'Total Number of Fans Age 13+ (View and/or Attend - add ,000)'
data2 = 'Year'  # Replace this with an actual Item name from your data

# Filter data for each item
item1_data = df_clean[df_clean['Item'] == data1].copy().sort_values('Year')
item2_data = df_clean[df_clean['Item'] == data2].copy().sort_values('Year')

# Create figure
fig = plt.figure()
ax = fig.add_subplot()

# Plot both items
ax.plot(item1_data['Year'], item1_data['Value'], color="black", 
        linestyle="dashed", marker='o', label=data1)
ax.plot(item2_data['Year'], item2_data['Value'], color="blue", 
        linestyle="dashed", marker='s', label=data2)

ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_title('Comparison Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Save to a specific location
save_path = r"C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Random_Plots/lds.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # ADD THIS LINE
plt.close()
print(f"Plot saved to: {save_path}")

# %% Time sereis
# MAKE SURE THIS GOES AT THE TOP OF EVERY PLOT
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

column = 'Total Number of Fans Age 13+ (View and/or Attend - add ,000)'

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(df[column], linestyle="dashed", label="Default")
ax.legend()

# IMPORTANT when using Agg:
plt.savefig("fans_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# %%