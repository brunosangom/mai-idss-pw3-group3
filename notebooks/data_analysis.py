# %% 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/Wildfire_Dataset.csv')
df.head()

# %%
unique_df = df.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)
len(unique_df)

# %%

# Load USA map
usa = gpd.read_file('../data/naturalearth_lowres/ne_110m_admin_0_countries.shp')
usa = usa[usa['NAME'] == 'United States of America']

# Plot base map
fig, ax = plt.subplots(figsize=(10, 8))
usa.plot(ax=ax, color='lightgray', edgecolor='black')

# Plot wildfire points
plt.scatter(unique_df['longitude'], unique_df['latitude'], s=10, c='red', alpha=0.5, label='Wildfires')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Wildfire Locations in USA')
plt.legend()
plt.show()

# %%
wildfire_counts = df['Wildfire'].value_counts()
print(wildfire_counts)

# %%
# Perform random train-test split (e.g., 70% train, 30% test)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Calculate percentage of 'No' and 'Yes' in each set and total
def wildfire_percentages(data):
    counts = data['Wildfire'].value_counts(normalize=True) * 100
    return counts

print("Total percentages:")
print(wildfire_percentages(df))

print("\nTrain set percentages:")
print(wildfire_percentages(train_df))

print("\nTest set percentages:")
print(wildfire_percentages(test_df))

# %%
# Count entries per latitude-longitude pair
pair_counts = df.groupby(['latitude', 'longitude']).size()

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(pair_counts, bins=range(1, pair_counts.max() + 2), edgecolor='black')
plt.xlabel('Number of Entries per Latitude-Longitude Pair')
plt.ylabel('Number of Pairs')
plt.title('Histogram of Entry Counts per Latitude-Longitude Pair')
plt.show()

# %%
pair_counts.describe()

# %%
