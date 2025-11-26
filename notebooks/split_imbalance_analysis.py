# %% [markdown]
# # Split Imbalance Analysis
# 
# This notebook investigates why there's a significant imbalance in wildfire rates
# between train/val and test splits.

# %%
import sys
sys.path.insert(0, '../src/backend')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import ExperimentConfig

# %%
# Load raw data
df = pd.read_csv('../data/Wildfire_Dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['Wildfire_binary'] = (df['Wildfire'] == 'Yes').astype(int)

print("=" * 80)
print("RAW DATA OVERVIEW")
print("=" * 80)
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Unique locations: {df.groupby(['latitude', 'longitude']).ngroup().nunique():,}")

# %%
# Analyze wildfire rate by year
df['year'] = df['datetime'].dt.year
yearly = df.groupby('year')['Wildfire_binary'].agg(['sum', 'count', 'mean']).round(4)
yearly.columns = ['wildfire_count', 'total_records', 'wildfire_rate']

print("\n" + "=" * 80)
print("WILDFIRE RATE BY YEAR")
print("=" * 80)
print(yearly)

# %%
# Simulate the chronological split per location (as done in dataset.py)
# Group by location, sort by datetime, and split 60/20/20

df['_group_id'] = df.groupby(['latitude', 'longitude']).ngroup()
df = df.sort_values(['_group_id', 'datetime']).reset_index(drop=True)
df['_group_idx'] = df.groupby('_group_id').cumcount()
group_sizes = df.groupby('_group_id').size()
df['_group_size'] = df['_group_id'].map(group_sizes)

train_ratio = 0.6
val_ratio = 0.2

df['_train_end'] = (df['_group_size'] * train_ratio).astype(int)
df['_val_end'] = (df['_group_size'] * (train_ratio + val_ratio)).astype(int)

df['_split'] = 'test'
df.loc[df['_group_idx'] < df['_train_end'], '_split'] = 'train'
df.loc[(df['_group_idx'] >= df['_train_end']) & 
       (df['_group_idx'] < df['_val_end']), '_split'] = 'val'

# %%
# Analyze which dates fall into each split
print("\n" + "=" * 80)
print("DATE RANGES BY SPLIT")
print("=" * 80)

for split in ['train', 'val', 'test']:
    split_df = df[df['_split'] == split]
    print(f"\n{split.upper()} Split:")
    print(f"  Records: {len(split_df):,}")
    print(f"  Date range: {split_df['datetime'].min()} to {split_df['datetime'].max()}")
    print(f"  Wildfire rate: {split_df['Wildfire_binary'].mean():.4f} ({split_df['Wildfire_binary'].mean()*100:.2f}%)")

# %%
# Analyze year distribution within each split
print("\n" + "=" * 80)
print("YEAR DISTRIBUTION PER SPLIT")
print("=" * 80)

for split in ['train', 'val', 'test']:
    split_df = df[df['_split'] == split]
    year_dist = split_df.groupby('year').agg({
        'Wildfire_binary': ['count', 'sum', 'mean']
    }).round(4)
    year_dist.columns = ['records', 'wildfires', 'rate']
    print(f"\n{split.upper()} Split Year Distribution:")
    print(year_dist)

# %%
# Visualize the temporal distribution of wildfires vs the split boundaries
print("\n" + "=" * 80)
print("MEDIAN DATE PER SPLIT")
print("=" * 80)

for split in ['train', 'val', 'test']:
    split_df = df[df['_split'] == split]
    median_date = split_df['datetime'].median()
    print(f"{split.upper()}: Median date = {median_date}")

# %%
# Check a few sample locations to understand the pattern
print("\n" + "=" * 80)
print("SAMPLE LOCATION ANALYSIS")
print("=" * 80)

# Get a few sample locations
sample_groups = df['_group_id'].drop_duplicates().head(5)

for group_id in sample_groups:
    group_df = df[df['_group_id'] == group_id].sort_values('datetime')
    lat, lon = group_df.iloc[0]['latitude'], group_df.iloc[0]['longitude']
    
    train_df = group_df[group_df['_split'] == 'train']
    val_df = group_df[group_df['_split'] == 'val']
    test_df = group_df[group_df['_split'] == 'test']
    
    print(f"\nLocation ({lat:.2f}, {lon:.2f}) - Group {group_id}:")
    print(f"  Total records: {len(group_df)}")
    
    if len(train_df) > 0:
        print(f"  Train: {train_df['datetime'].min().date()} to {train_df['datetime'].max().date()}, "
              f"Rate: {train_df['Wildfire_binary'].mean():.4f}")
    if len(val_df) > 0:
        print(f"  Val:   {val_df['datetime'].min().date()} to {val_df['datetime'].max().date()}, "
              f"Rate: {val_df['Wildfire_binary'].mean():.4f}")
    if len(test_df) > 0:
        print(f"  Test:  {test_df['datetime'].min().date()} to {test_df['datetime'].max().date()}, "
              f"Rate: {test_df['Wildfire_binary'].mean():.4f}")

# %%
# KEY INSIGHT: The problem
print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

# Calculate overall statistics
train_df = df[df['_split'] == 'train']
val_df = df[df['_split'] == 'val']
test_df = df[df['_split'] == 'test']

print("""
The chronological split strategy divides each location's time series as:
  - First 60% of timestamps -> Train
  - Next 20% of timestamps -> Val  
  - Last 20% of timestamps -> Test

This means:
  - Train contains mostly OLDER data (2014-2020)
  - Test contains mostly NEWER data (2022-2025)

The wildfire rate has been INCREASING over time:
  - 2014: 1.94%
  - 2020: 5.99%
  - 2024: 17.62%
  - 2025: 33.36%

This creates a TEMPORAL DISTRIBUTION SHIFT between splits!
The model trains on low-wildfire-rate data and tests on high-wildfire-rate data.
""")

# %%
# Quantify the temporal shift
train_years = train_df.groupby('year').size()
val_years = val_df.groupby('year').size()
test_years = test_df.groupby('year').size()

all_years_df = pd.DataFrame({
    'train': train_years,
    'val': val_years,
    'test': test_years
}).fillna(0).astype(int)

all_years_df['train_pct'] = (all_years_df['train'] / all_years_df['train'].sum() * 100).round(2)
all_years_df['val_pct'] = (all_years_df['val'] / all_years_df['val'].sum() * 100).round(2)
all_years_df['test_pct'] = (all_years_df['test'] / all_years_df['test'].sum() * 100).round(2)

print("\nRecords Distribution by Year and Split (%):")
print(all_years_df[['train_pct', 'val_pct', 'test_pct']])

# %%
print("\n" + "=" * 80)
print("PROPOSED SOLUTIONS")
print("=" * 80)

print("""
1. STRATIFIED TEMPORAL SPLIT:
   Instead of splitting each location chronologically, use a stratified approach
   that ensures similar wildfire rates across splits. This can be done by:
   a) Random sampling within temporal windows
   b) Stratified sampling based on wildfire occurrence

2. RANDOM SPLIT (with location grouping):
   Randomly assign entire locations to train/val/test to avoid temporal bias,
   while keeping each location's time series intact.

3. TEMPORAL STRATIFICATION:
   Split data such that each split contains proportional amounts from different
   time periods (e.g., each split gets 60% from early years, 20% mid, 20% late).

4. CLASS-BALANCED SAMPLING:
   During training, use weighted sampling or class weights to handle the
   imbalance, but this doesn't fix the distribution shift issue.

5. RE-SAMPLING THE TEST SET:
   Evaluate on a more representative test set that matches the training
   distribution (though this may hide real-world performance issues).

RECOMMENDATION: Option 2 (Random location-based split) or Option 3 (Temporal 
stratification) would best address the distribution shift while maintaining
the temporal structure needed for time-series prediction.
""")
