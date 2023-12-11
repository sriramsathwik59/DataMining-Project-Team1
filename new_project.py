#%%
import pandas as pd

# Specify the file path
file_path = 'Electric_Vehicle_Population_Size_History_By_County.csv'

# Load the dataset into a DataFrame
ev_data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(ev_data.head())

# %%
# Checking for null values in the dataset
null_values = ev_data.isnull().sum()
print(null_values)

# %%
# Removing rows with null values from the DataFrame
ev_data_cleaned = ev_data.dropna()

# Displaying the first few rows of the cleaned DataFrame
print(len(ev_data_cleaned))

# %%
ev_data_cleaned.info()
# %%
#divind sattes based on region
state_region_map = {
    'WA': 'West', 'OR': 'West', 'CA': 'West', 'AK': 'West', 'HI': 'West', 'NV': 'West', 'ID': 'West',
    'MT': 'West', 'WY': 'West', 'UT': 'West', 'AZ': 'West', 'CO': 'West', 'NM': 'West',
    'ND': 'Midwest', 'SD': 'Midwest', 'NE': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest',
    'WI': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'IN': 'Midwest', 'OH': 'Midwest',
    'PA': 'Northeast', 'NY': 'Northeast', 'VT': 'Northeast', 'NH': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast',
    'RI': 'Northeast', 'CT': 'Northeast', 'NJ': 'Northeast', 'DE': 'Northeast', 'MD': 'Northeast',
    'WV': 'South', 'VA': 'South', 'KY': 'South', 'TN': 'South', 'NC': 'South', 'SC': 'South', 'GA': 'South',
    'FL': 'South', 'AL': 'South', 'MS': 'South', 'AR': 'South', 'LA': 'South', 'TX': 'South', 'OK': 'South'
}

# Apply the mapping to the dataset to create the 'Region' column
ev_data_cleaned['Region'] = ev_data_cleaned['State'].map(state_region_map)

# Display the first few rows to confirm the addition of the 'Region' column
print(ev_data_cleaned.head())


# %%
# Saving the cleaned DataFrame with the 'Region' column to a new CSV file
output_file_path = 'dataset_new.csv'
ev_data_cleaned.to_csv(output_file_path, index=False)

# %%

#prerpocess the date column

import pandas as pd

# Load the updated dataset
updated_file_path = 'dataset_new.csv'
updated_ev_data = pd.read_csv(updated_file_path)

# Preprocess the 'Date' column by converting it to datetime format
updated_ev_data['Date'] = pd.to_datetime(updated_ev_data['Date'], errors='coerce')
# Replace the 'Date' column with just the year component
updated_ev_data['Date'] = updated_ev_data['Date'].dt.year.astype('Int64')  # Using 'Int64' to handle NaT (missing dates) properly

print(updated_ev_data.head())
#%%
# Subsetting the DataFrame to exclude rows where the state is "WA"
df_excluding_wa = updated_ev_data[updated_ev_data['State'] != 'WA']

# Displaying the first few rows of the subset to confirm the exclusion of "WA"
df_excluding_wa.head()

# Displaying the first few rows of the subset to confirm the exclusion of "WA"
df_excluding_wa.to_csv("update_new.csv",index=False)
# %%


df=pd.read_csv('update_new.csv')

# %%
df.head()
# %%

df.info()
# %%
# Converting the object type columns to their respective data types
# For 'County', 'State', 'Vehicle Primary Use', and 'Region' we will convert them to categorical type

df['County'] = df['County'].astype('category')
df['State'] = df['State'].astype('category')
df['Vehicle Primary Use'] = df['Vehicle Primary Use'].astype('category')
df['Region'] = df['Region'].astype('category')

# Displaying the updated DataFrame info to confirm the type conversion
df.info()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Visualization 1: Trend of Electric Vehicle Adoption Over Time
plt.figure(figsize=(12, 6))
updated_ev_data.groupby('Date')['Electric Vehicle (EV) Total'].sum().plot(kind='line', marker='o')
plt.title('Trend of Electric Vehicle Adoption Over Time')
plt.xlabel('Date')
plt.ylabel('Total Number of Electric Vehicles')
plt.xticks(rotation=45)
plt.tight_layout()

# Visualization 2: Regional Distribution of Electric Vehicles
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Electric Vehicle (EV) Total', data=df, estimator=sum, ci=None)
plt.title('Regional Distribution of Electric Vehicles')
plt.xlabel('Region')
plt.ylabel('Total Number of Electric Vehicles')
plt.tight_layout()

plt.show()
# %%
# Visualization: Trend of Electric Vehicles Over the Years by Region
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Electric Vehicle (EV) Total', hue='Region', marker='o')
plt.title('Trend of Electric Vehicles Over the Years by Region')
plt.xlabel('Year')
plt.ylabel('Total Electric Vehicles')
plt.legend(title='Region', loc='upper left')
plt.grid(True)
plt.show()
# %%
# Renaming the 'Date' column to 'Year'
df.rename(columns={'Date': 'Year'}, inplace=True)

# Displaying the first few rows of the DataFrame to confirm the column name change
df.head()

# %%
output_file_path = 'dataset_new2.csv'
df.to_csv(output_file_path, index=False)
# %%
df1=pd.read_csv('dataset_new2.csv')

df1.info()
# %%
df1['County'] = df1['County'].astype('category')
df1['State'] = df1['State'].astype('category')
df1['Vehicle Primary Use'] = df1['Vehicle Primary Use'].astype('category')
df1['Region'] = df1['Region'].astype('category')

# Displaying the updated DataFrame info to confirm the type conversion
df1.info()
# %%

# Assuming your DataFrame is named df1
region_subsets = {region: df1[df1['Region'] == region] for region in df1['Region'].unique()}

# To access the DataFrame for a specific region, use the region name as the key. For example:
south_df = region_subsets['South']
Northeast_df= region_subsets['Northeast']
West_df = region_subsets['West']
Midwest_df = region_subsets['Midwest']


# %%
south_df.info()
south_df.head()
# %%

# Creating a bar plot for the top 5 counties in the South region based on the total number of electric vehicles

# Data preparation
counties = ['Fairfax', 'Bexar', 'Norfolk', 'Prince William', 'Virginia Beach']
ev_totals = [608, 501, 337, 221, 218]

# Creating the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=counties, y=ev_totals, palette="viridis")
plt.title('Total Electric Vehicles in Top 5 Counties of the South Region')
plt.xlabel('County')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.show()



# %%
Northeast_df.head()
# %%
# Creating a subset for the Northeast region from the main DataFrame
northeast_df = df[df['Region'] == 'Northeast']

# Finding the top 5 counties in the Northeast region based on the total number of electric vehicles
top_5_counties_northeast_ev = northeast_df.groupby('County')['Electric Vehicle (EV) Total'].sum().nlargest(5)

# Extracting the county names and electric vehicle totals for the plot
northeast_counties = top_5_counties_northeast_ev.index.tolist()
northeast_ev_totals = top_5_counties_northeast_ev.values.tolist()

# Creating the bar plot for the top 5 counties in the Northeast region
plt.figure(figsize=(12, 6))
sns.barplot(x=northeast_counties, y=northeast_ev_totals, palette="viridis")
plt.title('Total Electric Vehicles in Top 5 Counties of the Northeast Region')
plt.xlabel('County')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.show()

# %%

# Filter data for the year 2023 and South region
data_2023_south = df1[(df1['Year'] == 2023) & (df1['Region'] == 'South')]

# Aggregate the total number of electric vehicles by county
county_ev_totals = data_2023_south.groupby('County')['Electric Vehicle (EV) Total'].sum()

# Sort the counties and select the top 5
top5_counties = county_ev_totals.sort_values(ascending=False).head(5)

# Display the top 5 counties
top5_counties

# %%
import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(10, 6))
top5_counties.plot(kind='bar', color='teal')
plt.title('Top 5 Counties in South Region by Total EVs in 2023')
plt.xlabel('County')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Filter data for the year 2023 and Northeast region
data_2023_northeast = df1[(df1['Year'] == 2023) & (df1['Region'] == 'Northeast')]

# Aggregate the total number of electric vehicles by county in the Northeast region
county_ev_totals_northeast = data_2023_northeast.groupby('County')['Electric Vehicle (EV) Total'].sum()

# Sort the counties in the Northeast region and select the top 5
top5_counties_northeast = county_ev_totals_northeast.sort_values(ascending=False).head(5)

# Plotting for Northeast region
plt.figure(figsize=(10, 6))
top5_counties_northeast.plot(kind='bar', color='skyblue')
plt.title('Top 5 Counties in Northeast Region by Total EVs in 2023')
plt.xlabel('County')
plt.ylabel('Total Electric Vehicles')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
plt.show()

# Return the top 5 counties data for further reference
top5_counties_northeast

# %%


#H0 The mean number of EVs in the top 5 counties of the South region is equal to or greater than the mean number of EVs in the top 5 counties of the Northeast region. This is our null hypothesis.
#H1: The mean number of EVs in the top 5 counties of the South region is less than the mean number of EVs in the top 5 counties of the Northeast region. This is our alternative hypothesis, which we are trying to test.


from scipy.stats import ttest_ind

# Means of the top 5 counties in both regions
mean_south = top5_counties.mean()
mean_northeast = top5_counties_northeast.mean()

# Perform a t-test
# Since we are doing a one-tailed test, we need to divide the p-value by 2
t_stat, p_value = ttest_ind(top5_counties, top5_counties_northeast, equal_var=False)
p_value_one_tailed = p_value / 2

mean_south, mean_northeast, t_stat, p_value_one_tailed 

'''
This implies that, based on our sample data, there is no statistically 
significant evidence to conclude that the mean number of EVs in the top 5 counties of the South
 region is less than that in the top 5 counties of the Northeast region.

'''

# %%



import matplotlib.pyplot as plt

# Grouping the data by state and summing up the BEVs and PHEVs
state_ev_totals = df1.groupby('State')[['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)']].sum()

# Sorting the states based on the total number of EVs (BEVs + PHEVs)
state_ev_totals['Total EVs'] = state_ev_totals['Battery Electric Vehicles (BEVs)'] + state_ev_totals['Plug-In Hybrid Electric Vehicles (PHEVs)']
state_ev_totals_sorted = state_ev_totals.sort_values(by='Total EVs', ascending=False)

# Plotting a stacked bar chart
state_ev_totals_sorted[['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)']].plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('Adoption of BEVs and PHEVs by State')
plt.xlabel('State')
plt.ylabel('Number of Electric Vehicles')
plt.legend(title='Vehicle Type')
plt.tight_layout()

# Display the plot
plt.show()

# %%
import statsmodels.api as sm

# Preparing the data for linear regression
# 'Year' and 'Total Vehicles' will be our independent variables (predictors)
# 'Electric Vehicle (EV) Total' is the dependent variable (target)

X = df1[['Year', 'Total Vehicles']]  # Independent variables
y = df1['Electric Vehicle (EV) Total']  # Dependent variable

# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Building the linear regression model
model = sm.OLS(y, X).fit()

# Getting the summary of the regression
model_summary = model.summary()

# Print the summary
print(model_summary)

# %%

df1.head()
# %%


# %%
'''
Task: Predict whether a county has a high or low adoption of electric vehicles based on the provided features.
Target Variable: Create a binary target variable (e.g., "High EV Adoption" vs. "Low EV Adoption") based on a threshold of Percent Electric Vehicles.
Logistic Regression: Use logistic regression to model the likelihood of high EV adoption based on the other features.

'''


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the threshold for high vs. low EV adoption (you can adjust this threshold)
threshold = 5.0  # For example, counties with >5% adoption are considered "high" adoption

# Create a binary target variable based on the threshold
df1['HighEVAdoption'] = (df1['Percent Electric Vehicles'] > threshold).astype(int)

# Define the predictors (features)
X = df1[['Year', 'Electric Vehicle (EV) Total', 'State', 'Region']]

# Define the target variable
y = df1['HighEVAdoption']

# Perform one-hot encoding for the categorical columns
X = pd.get_dummies(X, columns=['State', 'Region'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model (you can print accuracy, precision, recall, etc., as needed)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%

'''
Is there a significant difference in the average number of EVs among
different years (e.g., comparing 2017, 2019, and 2021)?
Are there significant variations in EV adoption rates among different states?

'''
# %%

'''
Null Hypothesis (H0): There is no significant difference in the average number of EVs among the years 2017, 2019, and 2021. 
This implies that any observed differences are due to random chance.
Alternative Hypothesis (H1): There is a significant difference in the
 average number of EVs among these years, indicating that changes over 
 the years are not just random variations.

'''
from scipy.stats import f_oneway
import pandas as pd
# Filter the data for the years 2017, 2019, and 2021
df_years = df1[df1['Year'].isin([2017, 2019, 2021])]

# Extract the EV totals for each of these years
data_2017 = df_years[df_years['Year'] == 2017]['Electric Vehicle (EV) Total']
data_2019 = df_years[df_years['Year'] == 2019]['Electric Vehicle (EV) Total']
data_2021 = df_years[df_years['Year'] == 2021]['Electric Vehicle (EV) Total']

# Perform ANOVA test
anova_result = f_oneway(data_2017, data_2019, data_2021)

# Output the result
print(anova_result)

# %%
import pandas as pd
from scipy.stats import chi2_contingency

'''
Null Hypothesis (H0): There is no association between the region and 
the vehicle primary use category in terms of EV adoption.
This means any observed association is due to random chance.
Alternative Hypothesis (H1): There is an association between the region and the vehicle primary use 
category in terms of EV adoption. This means the observed association is not due to random chance.

'''

contingency_table = pd.crosstab(df1['Region'], df1['Vehicle Primary Use'])

# Perform the Chi-Square Test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Output the results
print("Chi-square statistic:", chi2)
print("p-value:", p_value)
print("\nContingency Table:")
print(contingency_table)

# %%
import pandas as pd
import matplotlib.pyplot as plt

df_trucks = df1[df1['Vehicle Primary Use'] == 'Truck']
df_passengers = df1[df1['Vehicle Primary Use'] == 'Passenger']

# Summing up the BEVs and PHEVs for Trucks and Passengers
truck_bevs = df_trucks['Battery Electric Vehicles (BEVs)'].sum()
truck_phevs = df_trucks['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum()

passenger_bevs = df_passengers['Battery Electric Vehicles (BEVs)'].sum()
passenger_phevs = df_passengers['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum()

# Pie chart for Trucks
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie([truck_bevs, truck_phevs], labels=['BEVs', 'PHEVs'], autopct='%1.1f%%', startangle=140)
plt.title('Truck: BEVs vs PHEVs')

# Pie chart for Passengers
plt.subplot(1, 2, 2)
plt.pie([passenger_bevs, passenger_phevs], labels=['BEVs', 'PHEVs'], autopct='%1.1f%%', startangle=140)
plt.title('Passenger: BEVs vs PHEVs')

plt.show()


# %%
# Bar Chart of Average EVs by State
average_evs_by_state = df1.groupby('State')['Electric Vehicle (EV) Total'].mean().sort_values(ascending=False)

plt.figure(figsize=(15, 8))
average_evs_by_state.plot(kind='bar', color='skyblue')
plt.title('Average Number of EVs by State')
plt.xlabel('State')
plt.ylabel('Average Number of EVs')
plt.xticks(rotation=45)
plt.show()


# %%
# Bar Chart of EV Types by Region
bevs_by_region = df1.groupby('Region')['Battery Electric Vehicles (BEVs)'].sum()
phevs_by_region = df1.groupby('Region')['Plug-In Hybrid Electric Vehicles (PHEVs)'].sum()

# Stacking BEVs and PHEVs
plt.figure(figsize=(10, 6))
plt.bar(bevs_by_region.index, bevs_by_region, label='BEVs', color='blue', alpha=0.7)
plt.bar(phevs_by_region.index, phevs_by_region, bottom=bevs_by_region, label='PHEVs', color='orange', alpha=0.7)
plt.title('Comparison of BEVs and PHEVs by Region')
plt.xlabel('Region')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.show()





# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Load your dataset here
# df1 = pd.read_csv('path_to_your_dataset.csv')

# Assuming df1 is already loaded and preprocessed as before
county_region_grouped = df1.groupby(['Region', 'County']).agg({
    'Battery Electric Vehicles (BEVs)': 'sum', 
    'Plug-In Hybrid Electric Vehicles (PHEVs)': 'sum'
}).reset_index()

# Adding a total EVs column for sorting
county_region_grouped['Total EVs'] = county_region_grouped['Battery Electric Vehicles (BEVs)'] + county_region_grouped['Plug-In Hybrid Electric Vehicles (PHEVs)']

# Sorting to get the top 20 counties overall based on total EVs
top_20_counties_overall = county_region_grouped.sort_values(by='Total EVs', ascending=False).head(20)

# Extracting the unique regions from the top 20 counties
regions = top_20_counties_overall['Region'].unique()

# Assigning different colors for each region using NumPy
colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))
region_colors = {region: color for region, color in zip(regions, colors)}

# Creating the bar plot for the top 20 counties, colored by their region
plt.figure(figsize=(15, 8))
for _, row in top_20_counties_overall.iterrows():
    plt.bar(row['County'], row['Battery Electric Vehicles (BEVs)'], color=region_colors[row['Region']], edgecolor='white')
    plt.bar(row['County'], row['Plug-In Hybrid Electric Vehicles (PHEVs)'], bottom=row['Battery Electric Vehicles (BEVs)'], color=region_colors[row['Region']], alpha=0.5, edgecolor='white')

# Creating a custom legend
legend_patches = [mpatches.Patch(color=color, label=region) for region, color in region_colors.items()]
plt.legend(handles=legend_patches, title='Region')

plt.title('Top 20 Counties in EV Adoption (BEVs and PHEVs) by Region')
plt.xlabel('County')
plt.ylabel('Number of Vehicles')
plt.xticks(rotation=45)
plt.show()

# %%
