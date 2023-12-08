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
