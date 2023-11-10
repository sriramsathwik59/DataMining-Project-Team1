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

#Removing null

df.dropna()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Ensure the date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate data by Date
ev_trend = df.groupby('Date')['Electric Vehicle (EV) Total'].sum()

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
