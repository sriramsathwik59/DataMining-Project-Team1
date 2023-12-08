#%%
import pandas as pd
df=pd.read_csv("Electric_Vehicle_Population_Data.csv")
df = df.drop_duplicates()

df = df.dropna()


# %%
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
df
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model_counts = df['Make'] + ' ' + df['Model']
model_counts = model_counts.value_counts()
top_10_models = model_counts.head(10)

plt.figure(figsize=(12, 6))
top_10_models.plot(kind='bar')
plt.title('Top 10 Electric Vehicle Models')
plt.xlabel('Make and Model')
plt.ylabel('Count')
plt.show()

# %%
make_counts = df['Make'].value_counts().nlargest(5)

plt.figure(figsize=(12, 6))
sns.barplot(x=make_counts.values, y=make_counts.index, palette="viridis")
plt.title('Top 5 Makes of Electric Vehicles')
plt.xlabel('Count')
plt.ylabel('Make')
plt.show()
#%%
# 2. Bar chart for the count of Electric Vehicle Types
plt.figure(figsize=(10, 6))
df['Electric Vehicle Type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count of Electric Vehicle Types')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Count')
plt.show()

# 3. Scatter plot for Electric Range vs Model Year for Battery Electric Vehicles (BEV)
bev_data = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
plt.figure(figsize=(10, 6))
plt.scatter(bev_data['Model Year'], bev_data['Electric Range'], color='green', alpha=0.5)
plt.title('Electric Range vs Model Year for BEVs')
plt.xlabel('Model Year')
plt.ylabel('Electric Range')
plt.show()

# %%

from scipy.stats import ttest_ind



numeric_columns = df.select_dtypes(include=['number'])

# 3. Descriptive Statistics
desc_stats = numeric_columns.describe()

# 4. T-test Example (you can choose different columns based on your analysis)
column1 = 'Electric Range'
column2 = 'Model Year'

group1 = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)'][column1]
group2 = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)'][column1]

t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

# Print Descriptive Statistics
print("Descriptive Statistics:")
print(desc_stats)

# Print t-test results
print("\nT-test Results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")



# %%
# 2. Exclude non-numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# 4. Boxplot
# Visualize the distribution of electric range for each vehicle type
plt.figure(figsize=(12, 8))
sns.boxplot(x='Electric Vehicle Type', y='Electric Range', data=df)
plt.title("Boxplot of Electric Range by Electric Vehicle Type")
plt.show()
#%%
# 5. Countplot
# Visualize the count of each electric vehicle type
plt.figure(figsize=(8, 6))
sns.countplot(x='Electric Vehicle Type', data=df)
plt.title("Count of Electric Vehicle Types")
plt.show()
#%%
# 6. Bar chart
# Visualize the average electric range for each vehicle make
average_range_by_make = df.groupby('Make')['Electric Range'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
average_range_by_make.plot(kind='bar', color='skyblue')
plt.title("Average Electric Range by Vehicle Make")
plt.xlabel("Vehicle Make")
plt.ylabel("Average Electric Range")
plt.show()
#%%
# 7. Violin plot
# Visualize the distribution of electric range for each vehicle type
plt.figure(figsize=(12, 8))
sns.violinplot(x='Electric Vehicle Type', y='Electric Range', data=df, inner='quartile')
plt.title("Violin Plot of Electric Range by Electric Vehicle Type")
plt.show()

# %%
# %%
