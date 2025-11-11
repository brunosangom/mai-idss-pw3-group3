# %% 
import pandas as pd
import geopandas as gpd

df = pd.read_csv('../data/Wildfire_Dataset.csv')
df.head()

# %%
unique_df = df.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)

# %%
import matplotlib.pyplot as plt

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
