#%%
import pandas as pd
data=pd.read_csv("Electric_Vehicle_Population_Data.csv")
data = data.drop_duplicates()

data = data.dropna()

data.columns = data.columns.str.lower()
data['electric range'] = pd.to_numeric(data['electric range'], errors='coerce')

data = pd.get_dummies(data, columns=['electric vehicle type'])

# %%
print(data.isnull().sum())
data = data.dropna()
data = data.drop_duplicates()
data
print(data)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model_counts = data['make'] + ' ' + data['model']
model_counts = model_counts.value_counts()
top_10_models = model_counts.head(10)

plt.figure(figsize=(12, 6))
top_10_models.plot(kind='bar')
plt.title('Top 10 Electric Vehicle Models')
plt.xlabel('Make and Model')
plt.ylabel('Count')
plt.show()

# %%
make_counts = data['make'].value_counts().nlargest(5)

plt.figure(figsize=(12, 6))
sns.barplot(x=make_counts.values, y=make_counts.index, palette="viridis")
plt.title('Top 5 Makes of Electric Vehicles')
plt.xlabel('Count')
plt.ylabel('Make')
plt.show()
# %%
