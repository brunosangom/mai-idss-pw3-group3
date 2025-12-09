# %% [markdown]
# # Fire Onset Analysis
# 
# This notebook analyzes the frequency of fire onset events in the preprocessed dataset.
# Fire onset is defined as a transition from no-fire (0) to fire (1) in consecutive timesteps.

# %%
import sys
sys.path.append('../src/backend')

import pandas as pd
import numpy as np
from pathlib import Path
from config import ExperimentConfig

# %%
# Load configuration
config = ExperimentConfig('../src/backend/config.yaml')
data_config = config.get_data_config()

# %%
# Get cache paths
cache_key_params = {
    'data_path': data_config['path'],
    'features': sorted(data_config.get('features', [])),
    'split_ratios': data_config['split_ratios'],
    'window_size': data_config['window_size'],
    'temporal_bucket': data_config.get('temporal_bucket', 'sequential'),
}

import hashlib
import json
config_str = json.dumps(cache_key_params, sort_keys=True)
cache_key = hashlib.md5(config_str.encode()).hexdigest()[:12]

# Get absolute path to data directory (relative to notebooks directory)
notebook_dir = Path(__file__).parent.resolve()
project_root = notebook_dir.parent
cache_dir = project_root / "data" / "preprocessed_cache"

cache_paths = {
    'train': cache_dir / f"train_{cache_key}.csv",
    'val': cache_dir / f"val_{cache_key}.csv",
    'test': cache_dir / f"test_{cache_key}.csv",
}

print(f"Using cache key: {cache_key}")
print(f"Cache directory: {cache_dir}")

# %%
# Load preprocessed splits
dfs = []
for split_name, path in cache_paths.items():
    if path.exists():
        df = pd.read_csv(path)
        df['split'] = split_name
        dfs.append(df)
        print(f"Loaded {split_name}: {len(df):,} rows")
    else:
        print(f"Warning: {path} not found")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal combined rows: {len(combined_df):,}")

# %%
# Group by location and sort by time to identify fire onsets
combined_df = combined_df.sort_values(['latitude', 'longitude', 'datetime']).reset_index(drop=True)

# Create previous wildfire column within each location group
combined_df['prev_wildfire'] = combined_df.groupby(['latitude', 'longitude'])['Wildfire'].shift(1)

# First timestep of each location has no previous value - set to 0 (conservative assumption)
combined_df['prev_wildfire'] = combined_df['prev_wildfire'].fillna(0).astype(int)

# %%
# Identify fire onsets: previous was 0, current is 1
combined_df['is_onset'] = ((combined_df['prev_wildfire'] == 0) & (combined_df['Wildfire'] == 1)).astype(int)

# %%
# Calculate statistics
total_timesteps = len(combined_df)
total_onsets = combined_df['is_onset'].sum()
onset_percentage = (total_onsets / total_timesteps) * 100

print("\n" + "="*60)
print("FIRE ONSET ANALYSIS")
print("="*60)
print(f"Total timesteps: {total_timesteps:,}")
print(f"Fire onset events: {total_onsets:,}")
print(f"Fire onset percentage: {onset_percentage:.4f}%")
print("="*60)

# %%
# Break down by split
print("\nFire Onset Statistics by Split:")
print("-"*60)
for split in ['train', 'val', 'test']:
    split_df = combined_df[combined_df['split'] == split]
    split_total = len(split_df)
    split_onsets = split_df['is_onset'].sum()
    split_pct = (split_onsets / split_total) * 100 if split_total > 0 else 0
    
    print(f"{split.upper():5s}: {split_onsets:6,} onsets / {split_total:8,} timesteps = {split_pct:.4f}%")

# %%
# Additional context: overall wildfire rate
wildfire_rate = (combined_df['Wildfire'].sum() / total_timesteps) * 100
print("\n" + "-"*60)
print(f"Overall wildfire rate (any fire): {wildfire_rate:.2f}%")
print(f"Fire onset rate (0→1 transitions): {onset_percentage:.4f}%")
print(f"Onset events are {wildfire_rate/onset_percentage:.1f}x rarer than fire days")
print("-"*60)

# %%
