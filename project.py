#%%
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

# Example state abbreviation-to-region mapping for the U.S.
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


#%%
# Assuming your DataFrame is named df
df.to_csv('ev_final.csv', index=False)



#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd

# Aggregate data by Date
ev_trend = ev_final.groupby('Date')['Electric Vehicle (EV) Total'].sum()

# Plotting the trend
plt.figure(figsize=(12, 6))
plt.plot(ev_trend, marker='o', linestyle='-', color='b')
plt.title('Trend of Electric Vehicle Adoption Over Time')
plt.xlabel('Date')
plt.ylabel('Total Electric Vehicles')
plt.grid(True)
plt.show()

'''
There is a clear upward trend in the adoption of EVs. 
This indicates a growing interest and investment in electric vehicles over the time period
The most recent data points suggest that the growth in EV adoption has not plateaued
,indicating a continuing trend towards electric vehicle usage.
'''
#%%


# %%
import seaborn as sns

# Aggregate data by State
ev_by_state = df.groupby('State')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False)

# Plotting the distribution
plt.figure(figsize=(15, 8))
sns.barplot(x=ev_by_state.index, y=ev_by_state.values, palette="viridis")
plt.title('Distribution of Electric Vehicles by State')
plt.xlabel('State')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.show()

# %%
# Plotting the distribution with a log scale
plt.figure(figsize=(15, 8))
sns.barplot(x=ev_by_state.index, y=ev_by_state.values, palette="viridis")
plt.yscale('log')  # Setting the y-axis to a logarithmic scale
plt.title('Distribution of Electric Vehicles by State (Log Scale)')
plt.xlabel('State')
plt.ylabel('Total Electric Vehicles (Log Scale)')
plt.xticks(rotation=45)
plt.show()


'''
state like WA have significantly higher counts of electric vehicles. 
This suggests a strong geographic skew in EV adoption.

'''

# %%
import seaborn as sns

# Selecting the numerical columns for correlation analysis
numerical_cols = ['Battery Electric Vehicles (BEVs)', 
                  'Plug-In Hybrid Electric Vehicles (PHEVs)', 
                  'Electric Vehicle (EV) Total', 
                  'Non-Electric Vehicle Total', 
                  'Total Vehicles', 
                  'Percent Electric Vehicles']

# Calculating the correlation matrix
corr_matrix = df[numerical_cols].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Electric Vehicle Data')
plt.show()
'''
There are strong positive correlations between the counts of different types of electric vehicles (BEVs, PHEVs, and total EVs).
This suggests that regions with highnumbers of one type of EV are likely to have high
numbers of the other type as well.


'''
# %%
# Scatter plot for EVs vs Non-EVs
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Non-Electric Vehicle Total', y='Electric Vehicle (EV) Total', data=df)
plt.title('Relationship Between EVs and Non-EVs')
plt.xlabel('Non-Electric Vehicle Total')
plt.ylabel('Electric Vehicle (EV) Total')
plt.show()

# %%
# Proportions of BEVs and PHEVs
bevs = df['Battery Electric Vehicles (BEVs)'].sum()
phevs = df['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum()
plt.figure(figsize=(8, 8))
plt.pie([bevs, phevs], labels=['BEVs', 'PHEVs'], autopct='%1.1f%%')
plt.title('Proportion of BEVs and PHEVs')
plt.show()

# %%
#Research Question 1: Is there a significant difference in the adoption of BEVs and PHEVs?

#
#h0:There is no significant difference in the adoption of BEVs and PHEVs.
#h1:There is a significant difference in the adoption of BEVs and PHEVs.

from scipy import stats

# T-test between BEVs and PHEVs
t_statistic, p_value = stats.ttest_ind(df['Battery Electric Vehicles (BEVs)'], 
                                       df['Plug-In Hybrid Electric Vehicles (PHEVs)'])

# Interpretation of the result
if p_value < 0.05:
    result = "Reject the null hypothesis: There is a significant difference."
else:
    result = "Fail to reject the null hypothesis: No significant difference."

t_statistic, p_value, result


'''We reject the null hypothesis and conclude that there is a significant difference in the adoption of BEVs and PHEVs. 
This suggests that either BEVs or PHEVs are more prevalent,
indicating a preference or more favorable conditions for one type over the other '''


# %%

#Research Question 2: Is the proportion of electric vehicles different across vehicle primary uses?

#This question aims to understand if the primary use of a vehicle influences
# the likelihood of it being an electric vehicle.

#h0: The proportion of electric vehicles is the same across different primary uses.
#h1:The proportion of electric vehicles differs across different primary uses.

contingency_table = pd.crosstab(df['Vehicle Primary Use'], df['Electric Vehicle (EV) Total'] > 0)
chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

if p_val < 0.05:
    result = "Reject the null hypothesis: There is a difference in proportions."
else:
    result = "Fail to reject the null hypothesis: No significant difference in proportions."

chi2_stat, p_val, result
'''
 We reject the null hypothesis and conclude that there is a significant 
 difference in the proportions of electric vehicles across different primary uses. 
 This suggests that the likelihood of a vehicle being electric varies depending on its primary use.
   Some uses may be more conducive to or popular for electric vehicles than others.

'''



# %%
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map shapefile (built-in geopandas dataset)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the map with the electric vehicle data by country
merged_data = world.merge(df.groupby('State')['Electric Vehicle (EV) Total'].sum().reset_index(), 
                           how='left', left_on='iso_a3', right_on='State')

# Plotting the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_data.plot(column='Electric Vehicle (EV) Total', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# Add labels and title
ax.set_title('Electric Vehicle Adoption by Country', fontdict={'fontsize': '15', 'fontweight' : '3'})
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()



# %%
first_year = df['Date'].min().year
print(f"Starting Year: {first_year}")

last_year = df['Date'].max().year
print(f"Ending Year: {last_year}")

# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Assuming df is your DataFrame with the electric vehicle data
# Convert the 'Date' column to datetime format

# Extract relevant columns
data = df[['Date', 'State', 'Electric Vehicle (EV) Total']]

# Aggregate data by state and date
state_date_agg = data.groupby(['State', 'Date']).sum().reset_index()

# Visualize temporal trends for a specific state (e.g., 'WA' for Washington)
state_data = state_date_agg[state_date_agg['State'] == 'WA']

# Plot the temporal trends
plt.figure(figsize=(12, 6))
plt.plot(state_data['Date'], state_data['Electric Vehicle (EV) Total'], marker='o', linestyle='-', color='b')
plt.title('Temporal Trends of Electric Vehicle Adoption in Washington')
plt.xlabel('Date')
plt.ylabel('Total Electric Vehicles')
plt.grid(True)
plt.show()

# Time Series Forecasting with Exponential Smoothing
state_data = state_data.set_index('Date')
state_data.index.freq = 'D'  # Assuming daily frequency

# Instantiate Exponential Smoothing model
model = ExponentialSmoothing(state_data['Electric Vehicle (EV) Total'], trend='add', seasonal='add', seasonal_periods=365)

# Fit the model
fit_model = model.fit()

# Create a dataframe with future dates for forecasting
future_dates = pd.date_range(start=state_data.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
future = pd.DataFrame(index=future_dates)

# Generate forecasts
forecast = fit_model.forecast(steps=365)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(state_data.index, state_data['Electric Vehicle (EV) Total'], label='Historical Data')
plt.plot(future.index, forecast, label='Forecast', linestyle='--', color='r')
plt.title('Electric Vehicle Adoption Forecast in Washington')
plt.xlabel('Date')
plt.ylabel('Total Electric Vehicles')
plt.legend()
plt.show()


# %%

# %%
#Research Question 3: "How does the adoption of electric vehicles differ based on the primary use of the vehicle (e.g., passenger cars, trucks)?"

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame with the electric vehicle data
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate data by Vehicle Primary Use
ev_by_vehicle_type = df.groupby('Vehicle Primary Use')['Electric Vehicle (EV) Total'].sum().sort_values(ascending=False)

# Plotting the distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=ev_by_vehicle_type.index, y=ev_by_vehicle_type.values, palette="viridis")
plt.title('Distribution of Electric Vehicles by Vehicle Primary Use')
plt.xlabel('Vehicle Primary Use')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.show()



# %%
from scipy.stats import f_oneway

# Perform ANOVA test
anova_result = f_oneway(df[df['Vehicle Primary Use'] == 'Passenger']['Electric Vehicle (EV) Total'],
                        df[df['Vehicle Primary Use'] == 'Truck']['Electric Vehicle (EV) Total'])

# Print the ANOVA result
print("ANOVA Result:")
print(anova_result)

# Interpret the result
if anova_result.pvalue < 0.05:
    print("Reject the null hypothesis: There is a significant difference in means.")
else:
    print("Fail to reject the null hypothesis: No significant difference in means.")

# %%
#Model Building
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame with the electric vehicle data
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows with missing values for simplicity
df = df.dropna()

# Encode categorical variables (Vehicle Primary Use)
label_encoder = LabelEncoder()
df['Vehicle Primary Use'] = label_encoder.fit_transform(df['Vehicle Primary Use'])

# Select relevant features and target variable
X = df[['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)', 'Non-Electric Vehicle Total']]
y = df['Vehicle Primary Use']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame with the electric vehicle data
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows with missing values for simplicity
df = df.dropna()

# Encode categorical variables (Vehicle Primary Use)
label_encoder = LabelEncoder()
df['Vehicle Primary Use'] = label_encoder.fit_transform(df['Vehicle Primary Use'])

# Select relevant features and target variable
X = df[['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)', 'Non-Electric Vehicle Total']]
y = df['Vehicle Primary Use']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights to handle imbalanced data
class_weights = dict(zip(y.unique(), len(y) / (len(y.unique()) * y.value_counts())))

# Instantiate the decision tree classifier with class weights
model = DecisionTreeClassifier(random_state=42, class_weight=class_weights)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# 