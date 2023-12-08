import pandas as pd
import numpy as np


# %%
df=pd.read_csv("Electric_Vehicle_Population_Size_History_By_County.csv")

# %%
df
# %%
df.info()
# %%

# Check for any null values in each row
null_rows = df.isnull().any(axis=1)

if null_rows.any():
    print("There are rows with null fields.")

else:
    print("There are no rows with null fields.")

# %%
# Check for null values in each column
null_columns = df.isnull().sum()

if not null_columns.empty:
    print("Num of null value rows in the Columns are:")
    print(null_columns)
else:
    print("There are no columns with null values.")

#%%
#Removing null
df=df.dropna()
df
#%%
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.head()
#%%
from scipy.stats import zscore

numerical_cols = ['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)', 'Electric Vehicle (EV) Total', 'Non-Electric Vehicle Total', 'Total Vehicles', 'Percent Electric Vehicles']

z_scores = zscore(df[numerical_cols])

threshold = 3
outliers = (abs(z_scores) > threshold).any(axis=1)

if outliers.any():
    print("Rows with outliers:")
    
else:
    print("There are no rows with outliers.")

df[outliers]

#%%

# Count the unique states
unique_states_count = df['State'].nunique()

# Get the list of unique states
unique_states_list = df['State'].unique()

# Print the count and the list of unique states
print("Number of unique states in the dataset:", unique_states_count)
print("List of unique states:", unique_states_list)


#%%
#divide by regions
regions = {
    'South': ['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'MO', 'NC', 'SC', 'TN', 'TX', 'VA', 'WV'],
    'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY'],
    'North': ['CT', 'IL', 'IN', 'IA', 'KS', 'ME', 'MA', 'MI', 'MN', 'NE', 'NH', 'NJ', 'NY', 'ND', 'OH', 'PA', 'RI', 'SD', 'VT', 'WI'],
    'East': ['DE', 'MD', 'DC']
}

# Function to categorize each state
def categorize_state(state_abbr):
    for region, state_abbrs in regions.items():
        if state_abbr in state_abbrs:
            return region
    return 'Other'  # For states not in the mapping

# Create a copy of the DataFrame to avoid the SettingWithCopyWarning
df_copy = df.copy()

# Apply the function to the copy of your DataFrame
df_copy['Region'] = df_copy['State'].apply(categorize_state)

# Creating subsets using the copied DataFrame
south_df = df_copy[df_copy['Region'] == 'South']
west_df = df_copy[df_copy['Region'] == 'West']
north_df = df_copy[df_copy['Region'] == 'North']
east_df = df_copy[df_copy['Region'] == 'East']

# You can now work with south_df, west_df, north_df, and east_df as needed

#%%


ev_final=df_copy
ev_final.head()

len(ev_final)