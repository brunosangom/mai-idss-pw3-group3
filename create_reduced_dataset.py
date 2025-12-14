"""
Script to create a reduced version of the Wildfire_Dataset.csv
Includes the most recent time window with all locations and some true positives.
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Load the full dataset
print("Loading dataset...")
df = pd.read_csv('data/Wildfire_Dataset.csv')

print(f"Full dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for date/time columns
print(f"\nDataset info:")
print(df.info())

# Analyze the target variable distribution
if 'fire' in df.columns or 'Fire' in df.columns or 'target' in df.columns:
    target_col = 'fire' if 'fire' in df.columns else ('Fire' if 'Fire' in df.columns else 'target')
    print(f"\nTarget variable '{target_col}' distribution:")
    print(df[target_col].value_counts())
    print(f"Positive rate: {df[target_col].mean():.4f}")

# Find date columns
date_columns = []
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
        date_columns.append(col)
        print(f"\nUnique values in {col}: {df[col].nunique()}")
        print(f"Range: {df[col].min()} to {df[col].max()}")

# Check locations
location_columns = []
for col in df.columns:
    if any(loc in col.lower() for loc in ['lat', 'lon', 'location', 'country', 'region']):
        location_columns.append(col)
        print(f"\nUnique values in {col}: {df[col].nunique()}")

# Convert datetime to proper format
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

# Analyze temporal distribution
print(f"\n\n=== TEMPORAL ANALYSIS ===")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Check wildfire distribution
target_col = 'Wildfire'
df['fire_binary'] = (df[target_col] == 'Yes').astype(int)
print(f"\nOverall wildfire distribution:")
print(df[target_col].value_counts())
print(f"Positive rate: {df['fire_binary'].mean():.4f}")

# Analyze recent years
print(f"\n\n=== RECENT YEARS ANALYSIS ===")
recent_years = df[df['year'] >= 2020].copy()
print(f"Data from 2020+: {len(recent_years)} rows")
print(f"Positive rate in recent years: {recent_years['fire_binary'].mean():.4f}")

# Analyze by year and month
yearly_summary = df.groupby('year').agg({
    'fire_binary': ['sum', 'count', 'mean'],
    'latitude': 'nunique',
    'longitude': 'nunique'
}).round(4)
yearly_summary.columns = ['_'.join(col).strip() for col in yearly_summary.columns]
print(f"\n\nYearly summary (last 5 years):")
print(yearly_summary.tail(10))

# Find the most recent period with good fire coverage
print(f"\n\n=== FINDING OPTIMAL TIME WINDOW ===")
# Look at recent months with fires
recent_fires = df[df['fire_binary'] == 1].copy()
recent_fires = recent_fires.sort_values('datetime', ascending=False)
print(f"Most recent fires:")
print(recent_fires[['datetime', 'latitude', 'longitude', 'Wildfire']].head(20))

# Get last 6 months of data
latest_date = df['datetime'].max()
print(f"\nLatest date in dataset: {latest_date}")

# Try different time windows
for months_back in [3, 6, 9, 12]:
    cutoff_date = latest_date - pd.DateOffset(months=months_back)
    window_df = df[df['datetime'] >= cutoff_date]
    n_fires = window_df['fire_binary'].sum()
    n_locations = len(window_df.groupby(['latitude', 'longitude']))
    print(f"\nLast {months_back} months (from {cutoff_date.date()}):")
    print(f"  Rows: {len(window_df):,}")
    print(f"  Fires: {n_fires}")
    print(f"  Unique locations: {n_locations}")
    print(f"  Fire rate: {window_df['fire_binary'].mean():.4f}")
    print(f"  Size: ~{len(window_df) * 19 * 8 / (1024**2):.1f} MB")

# Create the reduced dataset - use last 12 months
print(f"\n\n=== CREATING REDUCED DATASET ===")
cutoff_date = latest_date - pd.DateOffset(months=12)
reduced_df = df[df['datetime'] >= cutoff_date].copy()

print(f"\nReduced dataset info:")
print(f"  Date range: {reduced_df['datetime'].min()} to {reduced_df['datetime'].max()}")
print(f"  Total rows: {len(reduced_df):,}")
print(f"  Total fires: {reduced_df['fire_binary'].sum()}")
print(f"  Fire rate: {reduced_df['fire_binary'].mean():.4f}")
print(f"  Unique locations: {len(reduced_df.groupby(['latitude', 'longitude']))}")
print(f"  Estimated size: ~{len(reduced_df) * 19 * 8 / (1024**2):.1f} MB")

# Drop the temporary columns
reduced_df = reduced_df.drop(columns=['year', 'month', 'fire_binary'])

# Verify we have all required columns
print(f"\nColumns in reduced dataset: {reduced_df.columns.tolist()}")

# Save the reduced dataset
output_path = 'data/Wildfire_Dataset_Reduced.csv'
print(f"\nSaving to {output_path}...")
reduced_df.to_csv(output_path, index=False)

print(f"\n✓ Successfully created reduced dataset!")
print(f"  Original size: ~1 GB ({len(df):,} rows)")
print(f"  Reduced size: ~36 MB ({len(reduced_df):,} rows)")
print(f"  Reduction: {(1 - len(reduced_df)/len(df))*100:.1f}%")
print(f"  File saved to: {output_path}")

# Verify the saved file
print(f"\nVerifying saved file...")
test_df = pd.read_csv(output_path, nrows=5)
print(f"Successfully loaded, first few rows:")
print(test_df.head())
