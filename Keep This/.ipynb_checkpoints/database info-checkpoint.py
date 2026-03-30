#%% connect to database
import sqlite3
import pandas as pd

# Path to your database
DB_PATH = r"//sdata/susers/tparker27/My Documents/GitHub/work_to_show/Arkansas Tech Social Media/social_media.db"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Close connection
conn.close() # must close connection or will not work 
#%% create a db from files 
import sqlite3
import pandas as pd
import os

# List of file paths
files = [
    r"C:/path/to/file1.csv",
    r"C:/path/to/file2.xlsx"
]

# Connect to SQLite (creates DB if it doesn't exist)
conn = sqlite3.connect(DB_PATH)

for file in files:
    # Use the filename (without extension) as the table name
    table_name = os.path.splitext(os.path.basename(file))[0]
    
    # Load file into pandas
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        print(f"Skipping unsupported file type: {file}")
        continue
    
    # Optional: clean column names (replace spaces with underscores)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    
    # Write to SQLite
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Loaded {file} into table '{table_name}'")

conn.close()
#%% get table names
import sqlite3
import pandas as pd

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Convert list of tuples to simple list
table_names = [t[0] for t in tables]

print(table_names)

# Close connection
conn.close() # must close connection or will not work 
#%% create a df from a table
conn = sqlite3.connect(DB_PATH)
career_outcomes = pd.read_sql_query('SELECT * FROM career_outcomes', conn)
career_outcomes.head()
# Close connection
conn.close() # must close connection or will not work 
#%% change table name 
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Dictionary of old table names → new table names
tables_to_rename = {
    'School of Business _Senior Exit Survey Results 2025': 'TABLE_25', 
    'School of Business Senior Exit Survey Fall 2023': 'TABLE_23'
}

for old_name, new_name in tables_to_rename.items():
    cursor.execute(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";')
    print(f'Renamed table "{old_name}" to "{new_name}"')

conn.commit()
conn.close()
#%% delete all from database

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Delete all rows from the 'students' table
cursor.execute("DELETE FROM survey_2023;")
conn.commit()

conn.close()

#%% delete a table 
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop tables one by one
tables_to_drop = ["survey_2025",
"survey_2023"]

for table in tables_to_drop:
    cursor.execute(f"DROP TABLE IF EXISTS {table};")
    print(f"Dropped table {table}")

conn.commit()
conn.close()
