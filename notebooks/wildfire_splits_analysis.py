# %% [markdown]
# # Wildfire Dataset Splits Analysis
# 
# This notebook analyzes the distribution of wildfire occurrences (Wildfire==Yes)
# in the total dataset and in each of the train/val/test splits.

# %%
import sys
sys.path.insert(0, '../src/backend')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from config import ExperimentConfig
from dataset import WildfireDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# Load configuration
config_path = '../src/backend/config.yaml'
config = ExperimentConfig(config_path)
config.get_data_config()['path'] = '../data/Wildfire_Dataset.csv'  # Ensure correct data path

print("Data Configuration:")
print(f"  - Data path: {config.get_data_config()['path']}")
print(f"  - Window size: {config.get_data_config()['window_size']}")
print(f"  - Split ratios: {config.get_data_config()['split_ratios']}")
print(f"  - Temporal bucket: {config.get_data_config().get('temporal_bucket', 'sequential')}")
print(f"  - Features: {config.get_data_config()['features']}")

# %%
# Create the dataset splits (like in the trainer)
train_dataset, val_dataset, test_dataset = WildfireDataset.create_splits(config)

print(f"\nDataset sizes (number of sequences):")
print(f"  - Train: {len(train_dataset):,}")
print(f"  - Val: {len(val_dataset):,}")
print(f"  - Test: {len(test_dataset):,}")
print(f"  - Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")

# %% [markdown]
# ## Analysis of Wildfire Occurrences per Timestep
# 
# For each sequence, we have `window_size` timesteps. We'll analyze how many 
# timesteps have Wildfire==Yes (label=1) in each split.

# %%
def analyze_wildfire_distribution(dataset, split_name):
    """
    Analyze the distribution of wildfire occurrences in a dataset split.
    
    Returns a dictionary with statistics.
    """
    if len(dataset) == 0:
        return {
            'split': split_name,
            'total_sequences': 0,
            'total_timesteps': 0,
            'wildfire_timesteps': 0,
            'no_wildfire_timesteps': 0,
            'wildfire_percentage': 0.0,
            'no_wildfire_percentage': 0.0
        }
    
    # Collect all labels
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())
    
    all_labels = np.concatenate(all_labels)
    
    total_timesteps = len(all_labels)
    wildfire_timesteps = np.sum(all_labels == 1)
    no_wildfire_timesteps = np.sum(all_labels == 0)
    
    return {
        'split': split_name,
        'total_sequences': len(dataset),
        'total_timesteps': total_timesteps,
        'wildfire_timesteps': int(wildfire_timesteps),
        'no_wildfire_timesteps': int(no_wildfire_timesteps),
        'wildfire_percentage': (wildfire_timesteps / total_timesteps) * 100,
        'no_wildfire_percentage': (no_wildfire_timesteps / total_timesteps) * 100
    }

# %%
# Analyze each split
train_stats = analyze_wildfire_distribution(train_dataset, 'Train')
val_stats = analyze_wildfire_distribution(val_dataset, 'Validation')
test_stats = analyze_wildfire_distribution(test_dataset, 'Test')

# Create summary DataFrame
stats_df = pd.DataFrame([train_stats, val_stats, test_stats])

# Calculate totals
total_timesteps = stats_df['total_timesteps'].sum()
total_wildfire = stats_df['wildfire_timesteps'].sum()
total_no_wildfire = stats_df['no_wildfire_timesteps'].sum()

total_stats = {
    'split': 'TOTAL',
    'total_sequences': stats_df['total_sequences'].sum(),
    'total_timesteps': total_timesteps,
    'wildfire_timesteps': total_wildfire,
    'no_wildfire_timesteps': total_no_wildfire,
    'wildfire_percentage': (total_wildfire / total_timesteps) * 100 if total_timesteps > 0 else 0,
    'no_wildfire_percentage': (total_no_wildfire / total_timesteps) * 100 if total_timesteps > 0 else 0
}

stats_df = pd.concat([stats_df, pd.DataFrame([total_stats])], ignore_index=True)

# %%
# Display results
print("=" * 80)
print("WILDFIRE DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\n### Summary Statistics ###\n")
print(stats_df.to_string(index=False))

# %%
# Detailed per-split analysis
print("\n" + "=" * 80)
print("DETAILED PER-SPLIT ANALYSIS")
print("=" * 80)

for stats in [train_stats, val_stats, test_stats]:
    print(f"\n### {stats['split']} Split ###")
    print(f"  Total sequences: {stats['total_sequences']:,}")
    print(f"  Total timesteps: {stats['total_timesteps']:,}")
    print(f"  Wildfire=Yes timesteps: {stats['wildfire_timesteps']:,} ({stats['wildfire_percentage']:.4f}%)")
    print(f"  Wildfire=No timesteps: {stats['no_wildfire_timesteps']:,} ({stats['no_wildfire_percentage']:.4f}%)")

print(f"\n### Overall (All Splits Combined) ###")
print(f"  Total sequences: {total_stats['total_sequences']:,}")
print(f"  Total timesteps: {total_stats['total_timesteps']:,}")
print(f"  Wildfire=Yes timesteps: {total_stats['wildfire_timesteps']:,} ({total_stats['wildfire_percentage']:.4f}%)")
print(f"  Wildfire=No timesteps: {total_stats['no_wildfire_timesteps']:,} ({total_stats['no_wildfire_percentage']:.4f}%)")

# %% [markdown]
# ## Analysis of Last Timestep Only
# 
# Since the model predicts based on the last timestep of each sequence,
# let's also analyze the distribution considering only the last timestep.

# %%
def analyze_last_timestep_distribution(dataset, split_name):
    """
    Analyze the distribution of wildfire occurrences considering only
    the last timestep of each sequence (which is what the model predicts).
    """
    if len(dataset) == 0:
        return {
            'split': split_name,
            'total_samples': 0,
            'wildfire_samples': 0,
            'no_wildfire_samples': 0,
            'wildfire_percentage': 0.0,
            'no_wildfire_percentage': 0.0
        }
    
    # Collect only the last timestep label from each sequence
    last_labels = []
    for _, labels in dataset:
        last_labels.append(labels[-1].item())
    
    last_labels = np.array(last_labels)
    
    total_samples = len(last_labels)
    wildfire_samples = np.sum(last_labels == 1)
    no_wildfire_samples = np.sum(last_labels == 0)
    
    return {
        'split': split_name,
        'total_samples': total_samples,
        'wildfire_samples': int(wildfire_samples),
        'no_wildfire_samples': int(no_wildfire_samples),
        'wildfire_percentage': (wildfire_samples / total_samples) * 100,
        'no_wildfire_percentage': (no_wildfire_samples / total_samples) * 100
    }

# %%
# Analyze last timestep for each split
train_last = analyze_last_timestep_distribution(train_dataset, 'Train')
val_last = analyze_last_timestep_distribution(val_dataset, 'Validation')
test_last = analyze_last_timestep_distribution(test_dataset, 'Test')

# Create summary DataFrame
last_df = pd.DataFrame([train_last, val_last, test_last])

# Calculate totals
total_samples = last_df['total_samples'].sum()
total_wildfire_last = last_df['wildfire_samples'].sum()
total_no_wildfire_last = last_df['no_wildfire_samples'].sum()

total_last = {
    'split': 'TOTAL',
    'total_samples': total_samples,
    'wildfire_samples': total_wildfire_last,
    'no_wildfire_samples': total_no_wildfire_last,
    'wildfire_percentage': (total_wildfire_last / total_samples) * 100 if total_samples > 0 else 0,
    'no_wildfire_percentage': (total_no_wildfire_last / total_samples) * 100 if total_samples > 0 else 0
}

last_df = pd.concat([last_df, pd.DataFrame([total_last])], ignore_index=True)

# %%
print("\n" + "=" * 80)
print("LAST TIMESTEP ANALYSIS (Model Prediction Target)")
print("=" * 80)

print("\n### Summary Statistics (Last Timestep Only) ###\n")
print(last_df.to_string(index=False))

print("\n### Detailed Analysis ###")
for stats in [train_last, val_last, test_last]:
    print(f"\n{stats['split']} Split:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Wildfire=Yes: {stats['wildfire_samples']:,} ({stats['wildfire_percentage']:.4f}%)")
    print(f"  Wildfire=No: {stats['no_wildfire_samples']:,} ({stats['no_wildfire_percentage']:.4f}%)")

print(f"\nOverall (All Splits):")
print(f"  Total samples: {total_last['total_samples']:,}")
print(f"  Wildfire=Yes: {total_last['wildfire_samples']:,} ({total_last['wildfire_percentage']:.4f}%)")
print(f"  Wildfire=No: {total_last['no_wildfire_samples']:,} ({total_last['no_wildfire_percentage']:.4f}%)")

# %% [markdown]
# ## Class Imbalance Summary
# 
# The analysis shows the class imbalance in the wildfire prediction task.

# %%
print("\n" + "=" * 80)
print("CLASS IMBALANCE SUMMARY")
print("=" * 80)

imbalance_ratio = total_no_wildfire_last / total_wildfire_last if total_wildfire_last > 0 else float('inf')

print(f"\nClass imbalance ratio (No Wildfire : Wildfire): {imbalance_ratio:.2f}:1")
print(f"This means for every wildfire event, there are approximately {imbalance_ratio:.1f} non-wildfire events.")

if imbalance_ratio > 10:
    print("\n⚠️  HIGH CLASS IMBALANCE detected!")
    print("   Consider using techniques like:")
    print("   - Weighted loss function")
    print("   - Oversampling (SMOTE)")
    print("   - Threshold tuning (already enabled in config)")
    print("   - Focal loss")
elif imbalance_ratio > 3:
    print("\n⚡ MODERATE CLASS IMBALANCE detected.")
    print("   The threshold tuning in the config should help address this.")
else:
    print("\n✅ Class distribution is relatively balanced.")

#%%
# Load the dataset
df = pd.read_csv(config.get_data_config()['path'])

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter only wildfire occurrences (Wildfire == 'Yes')
wildfires = df[df['Wildfire'] == 'Yes']

# Count wildfires per date
wildfire_counts = wildfires.groupby('datetime').size().reset_index(name='wildfire_count')

# Plot
plt.figure(figsize=(14, 6))
plt.plot(wildfire_counts['datetime'], wildfire_counts['wildfire_count'], linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Wildfire Count')
plt.title('Total Wildfires Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wildfire_over_time.png', dpi=150)
print('Plot saved to wildfire_over_time.png')
print(f'Date range: {wildfire_counts["datetime"].min()} to {wildfire_counts["datetime"].max()}')
print(f'Total dates with wildfires: {len(wildfire_counts)}')
# %%
