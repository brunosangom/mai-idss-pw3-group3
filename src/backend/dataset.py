
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import logging
import hashlib
import json
import sys
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

class WildfireDataset(Dataset):

    # Default cache directory relative to data path
    CACHE_DIR = "data/preprocessed_cache"

    def __init__(self, config, split='train', _preprocessed_df=None, _feature_cols=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.split = split
        self.data_config = self.config.get_data_config()
        self.window_size = self.data_config['window_size']

        if _preprocessed_df is not None:
            # Use pre-processed data from factory method
            self.df = _preprocessed_df
            self._create_sequences(_feature_cols)
        else:
            raise ValueError("WildfireDataset must be created via the create_splits factory method.")

    @classmethod
    def _get_cache_key(cls, config):
        """
        Generate a unique cache key based on configuration parameters that affect preprocessing.
        """
        data_config = config.get_data_config()
        
        # Include all parameters that affect the preprocessing output
        cache_params = {
            'data_path': data_config['path'],
            'features': sorted(data_config.get('features', [])),
            'split_ratios': data_config['split_ratios'],
            'window_size': data_config['window_size'],
        }
        
        # Create a hash of the configuration
        config_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    @classmethod
    def _get_cache_paths(cls, config):
        """
        Get the cache file paths for each split.
        
        Returns:
            dict: Dictionary with 'train', 'val', 'test' keys and file paths as values
        """
        cache_key = cls._get_cache_key(config)
        cache_dir = Path(cls.CACHE_DIR)
        
        return {
            'train': cache_dir / f"train_{cache_key}.csv",
            'val': cache_dir / f"val_{cache_key}.csv",
            'test': cache_dir / f"test_{cache_key}.csv",
            'feature_cols': cache_dir / f"feature_cols_{cache_key}.json",
        }

    @classmethod
    def _cache_exists(cls, config):
        """
        Check if cached preprocessed data exists for the given configuration.
        """
        cache_paths = cls._get_cache_paths(config)
        return all(Path(p).exists() for p in cache_paths.values())

    @classmethod
    def _save_to_cache(cls, config, df, feature_cols):
        """
        Save preprocessed dataframe and feature columns to cache.
        """
        logger = logging.getLogger(__name__)
        cache_paths = cls._get_cache_paths(config)
        cache_dir = Path(cls.CACHE_DIR)
        
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split to a separate CSV
        for split in ['train', 'val', 'test']:
            split_df = df[df['_split'] == split]
            split_df.to_csv(cache_paths[split], index=False)
            logger.info(f"Saved {split} split ({len(split_df)} rows) to {cache_paths[split]}")
        
        # Save feature columns
        with open(cache_paths['feature_cols'], 'w') as f:
            json.dump(feature_cols, f)
        
        logger.info(f"Preprocessed data cached with key: {cls._get_cache_key(config)}")

    @classmethod
    def _load_from_cache(cls, config):
        """
        Load preprocessed dataframe and feature columns from cache.
        
        Returns:
            tuple: (combined_df, feature_cols)
        """
        logger = logging.getLogger(__name__)
        cache_paths = cls._get_cache_paths(config)
        
        logger.info(f"Loading preprocessed data from cache (key: {cls._get_cache_key(config)})")
        
        # Load each split and combine
        dfs = []
        for split in ['train', 'val', 'test']:
            split_df = pd.read_csv(cache_paths[split])
            split_df['Wildfire'] = split_df['Wildfire'].astype(np.int32)
            logger.info(f"Loaded {split} split ({len(split_df)} rows) from cache")
            dfs.append(split_df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Load feature columns
        with open(cache_paths['feature_cols'], 'r') as f:
            feature_cols = json.load(f)
        
        return combined_df, feature_cols

    @classmethod
    def create_splits(cls, config):
        """
        Factory method to create train, val, and test datasets.
        Loads CSV once and performs common preprocessing only once.
        Uses caching to avoid recomputing preprocessing if cached data exists.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        logger = logging.getLogger(__name__)
        data_config = config.get_data_config()
        window_size = data_config['window_size']
        
        # Check if cached data exists
        if cls._cache_exists(config):
            logger.info("Found cached preprocessed data, loading from cache...")
            df, feature_cols = cls._load_from_cache(config)
            
            # Create datasets for each split using cached data
            train_dataset = cls(config, split='train', _preprocessed_df=df, _feature_cols=feature_cols)
            val_dataset = cls(config, split='val', _preprocessed_df=df, _feature_cols=feature_cols)
            test_dataset = cls(config, split='test', _preprocessed_df=df, _feature_cols=feature_cols)
            
            return train_dataset, val_dataset, test_dataset
        
        logger.info("No cached data found, running preprocessing pipeline...")
        
        # Load data once
        logger.info(f"Loading data from {data_config['path']}")
        required_cols = ["latitude", "longitude", "datetime", "Wildfire"]
        features = data_config.get('features', [])
        all_cols = list(set(required_cols + features))
        
        df = pd.read_csv(data_config['path'], usecols=all_cols)
        df = df.dropna(subset=['latitude', 'longitude', 'datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Wildfire'] = df['Wildfire'].map({'Yes': 1, 'No': 0})
        df['Wildfire'] = df['Wildfire'].astype(np.int32)
        logger.debug(f"Loaded dataframe with shape {df.shape}")
        
        # Perform common preprocessing once
        logger.info("Preprocessing data for all splits")
        
        train_ratio = data_config['split_ratios']['train']
        val_ratio = data_config['split_ratios']['val']
        
        # Normalize lat/lon globally
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        df['latitude_norm'] = (df['latitude'] - lat_min) / (lat_max - lat_min)
        df['longitude_norm'] = (df['longitude'] - lon_min) / (lon_max - lon_min)
        df['datetime_norm'] = df['datetime'].dt.dayofyear / 365.0
        
        # Normalize year based on min-max
        year_min, year_max = df['datetime'].dt.year.min(), df['datetime'].dt.year.max()
        if year_max > year_min:
            df['year_norm'] = (df['datetime'].dt.year - year_min) / (year_max - year_min)
        else:
            df['year_norm'] = 0.0
        
        # Identify numeric features for normalization
        exclude_cols = {'latitude', 'longitude', 'datetime', 'Wildfire', 
                        'datetime_norm', 'latitude_norm', 'longitude_norm', 'year_norm'}
        numeric_features = [col for col in df.select_dtypes(include=np.number).columns 
                           if col not in exclude_cols]
        
        # Create group identifier
        df['_group_id'] = df.groupby(['latitude', 'longitude']).ngroup()
        df = df.sort_values(['_group_id', 'datetime']).reset_index(drop=True)
        
        # Add within-group index for split calculation
        df['_group_idx'] = df.groupby('_group_id').cumcount()
        group_sizes = df.groupby('_group_id').size()
        df['_group_size'] = df['_group_id'].map(group_sizes)
        
        # Calculate split boundaries per row
        df['_train_end'] = (df['_group_size'] * train_ratio).astype(int)
        df['_val_end'] = (df['_group_size'] * (train_ratio + val_ratio)).astype(int)
        
        # Determine which split each row belongs to
        df['_split'] = 'test'
        df.loc[df['_group_idx'] < df['_train_end'], '_split'] = 'train'
        df.loc[(df['_group_idx'] >= df['_train_end']) & 
                    (df['_group_idx'] < df['_val_end']), '_split'] = 'val'
        
        # Compute normalization stats per group using train+val data
        train_val_mask = df['_split'].isin(['train', 'val'])
        
        groups_to_drop = set()
        constant_column_counts = defaultdict(int)
        
        if numeric_features:
            group_stats = df[train_val_mask].groupby('_group_id')[numeric_features].agg(['mean', 'std'])
            group_stats.columns = ['_'.join(col) for col in group_stats.columns]
            
            std_cols = [f'{col}_std' for col in numeric_features]
            all_constant_mask = (group_stats[std_cols] == 0).all(axis=1)
            groups_to_drop = set(group_stats[all_constant_mask].index)
            
            if groups_to_drop:
                logger.info(f"Dropping {len(groups_to_drop)} groups with all constant features")
                df = df[~df['_group_id'].isin(groups_to_drop)]
                group_stats = group_stats[~all_constant_mask]
            
            total_groups = df['_group_id'].nunique()
            
            for col in numeric_features:
                mean_col = f'{col}_mean'
                std_col = f'{col}_std'
                
                df[f'_{mean_col}'] = df['_group_id'].map(group_stats[mean_col])
                df[f'_{std_col}'] = df['_group_id'].map(group_stats[std_col])
                
                constant_groups = (group_stats[std_col] == 0).sum()
                if constant_groups > 0:
                    constant_column_counts[col] = constant_groups
                
                std_values = df[f'_{std_col}']
                mean_values = df[f'_{mean_col}']
                
                df[col] = np.where(
                    std_values > 0,
                    (df[col] - mean_values) / std_values,
                    0
                )
                
                df.drop(columns=[f'_{mean_col}', f'_{std_col}'], inplace=True)
        else:
            total_groups = df['_group_id'].nunique()
        
        # Log summary table of constant columns
        if constant_column_counts:
            summary_lines = [
                "\n" + "=" * 60,
                "Constant Column Summary (columns with std=0 set to zero)",
                "=" * 60,
                f"{'Feature':<30} {'Count':>10} {'Percentage':>15}",
                "-" * 60
            ]
            for col, count in sorted(constant_column_counts.items()):
                percentage = (count / total_groups) * 100
                summary_lines.append(f"{col:<30} {count:>10} {percentage:>14.2f}%")
            summary_lines.append("=" * 60 + "\n")
            logger.info("\n".join(summary_lines))
        
        # Define feature columns for sequences
        meta_cols = {'latitude', 'longitude', 'datetime', 'Wildfire', '_group_id', 
                     '_group_idx', '_group_size', '_train_end', '_val_end', '_split'}
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        # Save preprocessed data to cache for future runs
        cls._save_to_cache(config, df, feature_cols)
        
        # Create datasets for each split
        train_dataset = cls(config, split='train', _preprocessed_df=df, _feature_cols=feature_cols)
        val_dataset = cls(config, split='val', _preprocessed_df=df, _feature_cols=feature_cols)
        test_dataset = cls(config, split='test', _preprocessed_df=df, _feature_cols=feature_cols)
        
        return train_dataset, val_dataset, test_dataset

    def _create_sequences(self, feature_cols):
        """Create sequences for the current split from preprocessed data."""
        # Process only the required split
        split_df = self.df[self.df['_split'] == self.split].copy()
        
        self.logger.info(f"Creating sequences for {split_df['_group_id'].nunique()} groups in split '{self.split}'")
        
        # Create sequences using optimized approach
        sequences = []
        labels = []
        metadata = []  # Store (latitude, longitude, datetime) for each sequence's last time step
        
        # Group the split data and process efficiently
        grouped = split_df.groupby('_group_id')
        
        for group_id, group in tqdm(grouped, desc=f"Creating sequences ({self.split})", file=sys.stdout):
            n = len(group)
            if n < self.window_size:
                continue
            
            # Extract feature values and labels as numpy arrays (single conversion)
            features = group[feature_cols].values.astype(np.float32)
            labels_arr = group['Wildfire'].values
            
            # Extract metadata columns
            lat_arr = group['latitude'].values
            lon_arr = group['longitude'].values
            datetime_arr = group['datetime'].values
            
            # Use sliding window view for efficient sequence creation (no copy!)
            num_sequences = n - self.window_size + 1
            
            # Create views using stride tricks
            feature_windows = sliding_window_view(features, window_shape=(self.window_size, len(feature_cols)))
            feature_windows = feature_windows.squeeze(axis=1)  # Remove extra dimension
            
            label_windows = sliding_window_view(labels_arr, window_shape=self.window_size)
            
            # Convert all sequences at once to tensors
            for i in range(num_sequences - 1):  # -1 to match original behavior
                sequences.append(torch.from_numpy(feature_windows[i].copy()))
                labels.append(torch.from_numpy(label_windows[i].copy()))
                # Store metadata for the last time step of this sequence (index i + window_size - 1)
                last_idx = i + self.window_size - 1
                metadata.append((lat_arr[last_idx], lon_arr[last_idx], datetime_arr[last_idx]))
        
        self.logger.debug(f"Created {len(sequences)} sequences for split {self.split}")
        self.sequences = sequences
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_metadata(self, idx):
        """Get metadata (latitude, longitude, datetime) for a sequence."""
        return self.metadata[idx]

    def get_all_metadata(self):
        """Get all metadata as a list of tuples (latitude, longitude, datetime)."""
        return self.metadata

    def get_num_features(self):
        if len(self.sequences) == 0:
            # This is a fallback for an empty dataset.
            # We need to calculate the number of features based on the config.
            required_cols = ["latitude", "longitude", "datetime", "Wildfire"]
            features = self.data_config.get('features', [])

            all_cols = list(set(required_cols + features))
            return len(all_cols) - 1  # Exclude 'Wildfire' label
        return self.sequences[0].shape[1]
