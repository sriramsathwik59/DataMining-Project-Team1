#%%
import pandas as pd
df=pd.read_csv("Electric_Vehicle_Population_Data.csv")
df = df.iloc[:, 1:]


# %%
print(df.isnull().sum())
df = df.dropna()
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

#%%


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have the data in a CSV file named 'ev_data.csv'
data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Filter the data to include only BEVs and PHEVs
bev_phev_data = data[(data['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)') | 
                     (data['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)')]

# Group data by Model Year and count the number of EVs
year_counts = bev_phev_data.groupby('Model Year').size()

# Plot adoption trends by year
plt.figure(figsize=(10, 6))
year_counts.plot(kind='bar', color='skyblue')
plt.title('EV Adoption Trends in Washington State by Year')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt



# Filter the data to include only BEVs and PHEVs
bev_phev_data = data[(data['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)') | 
                     (data['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)')]

# Group data by County and count the number of EVs
county_counts = bev_phev_data['County'].value_counts()

# Plot adoption patterns by county
plt.figure(figsize=(12, 6))
county_counts.plot(kind='bar', color='skyblue')
plt.title('EV Adoption by County in Washington State')
plt.xlabel('County')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()

# Group data by City and count the number of EVs
city_counts = bev_phev_data['City'].value_counts()

# Plot adoption patterns by city
plt.figure(figsize=(12, 6))
city_counts.head(20).plot(kind='bar', color='salmon')
plt.title('Top 20 Cities with EV Adoption in Washington State')
plt.xlabel('City')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt



# Filter the data to include only BEVs and PHEVs
bev_phev_data = data[(data['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)') | 
                     (data['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)')]

# Group data by County and count the number of EVs
county_counts = bev_phev_data['County'].value_counts().head(25)

# Create a dictionary to map counties to their cities
county_to_city = {}
for county, city in zip(data['County'], data['City']):
    county_to_city[county] = city

# Create a new DataFrame with counties and corresponding cities
county_data = pd.DataFrame({'County': county_counts.index, 'City': county_counts.index.map(county_to_city), 'EV Count': county_counts.values})

# Plot adoption patterns by county and label with cities
plt.figure(figsize=(12, 8))
plt.bar(county_data['County'], county_data['EV Count'], color='skyblue')
plt.title('Top 25 Counties with EV Adoption in Washington State')
plt.xlabel('County (City)')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45, ha='right')

# Add labels with city names above the bars
for i, row in county_data.iterrows():
    plt.text(i, row['EV Count'] + 10, row['City'], rotation=45, ha='right')

plt.show()

# %%


# %%

# Filter the data to include only BEVs and PHEVs
bev_phev_data = data[(data['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)') | 
                     (data['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)')]

# Replace specific values in the "CAFV Eligibility" column in the DataFrame
bev_phev_data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].replace(
    {
        "Clean Alternative Fuel Vehicle Eligible": "Eligible",
        "Not eligible due to low battery range": "Not eligible"
    },
    inplace=True
)

# Group data by CAFV Eligibility and count the number of vehicles for each category
cafv_eligibility_counts = bev_phev_data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()

# Create a bar chart to visualize CAFV Eligibility categories
plt.figure(figsize=(10, 6))
cafv_eligibility_counts.plot(kind='bar', color='lightgreen')
plt.title('CAFV Eligibility of Electric Vehicles')
plt.xlabel('CAFV Eligibility')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()


# %%
data.head()

# %%
unique_caf = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique()
print(unique_caf)

# %%
# Replace values in the specified column
df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].replace({
    'Clean Alternative Fuel Vehicle Eligible': 'Eligible',
    'Eligibility unknown as battery range has not been researched': 'Unknown',
    'Not eligible due to low battery range': 'Not Eligible'
})

# Print the updated unique values in the column
print(df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique())


# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Electric Vehicle Type', axis=1)  
y = df['Electric Vehicle Type']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))

coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
print(coefficients)
# %%
