
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class WildfireDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split # @todo: avoid redundant computation by implementing caching or returning all splits at once
        self.data_config = self.config.get_data_config()
        self.training_config = self.config.get_training_config()
        self.window_size = self.training_config['window_size']

        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        required_cols = ["latitude", "longitude", "datetime", "Wildfire"]
        features = self.data_config.get('features', [])

        all_cols = list(set(required_cols + features))
        
        df = pd.read_csv(self.data_config['path'])
        # Drop rows with missing values in key columns
        df = df.dropna(subset=['latitude', 'longitude', 'datetime'])

        # Convert 'datetime' column to pandas datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])

        self.df = df[all_cols + ['datetime']]


    def _preprocess_data(self):
        # Group by location
        grouped = self.df.groupby(['latitude', 'longitude'])
        
        self.sequences = []
        self.labels = []

        # Normalize lat/lon before splitting
        lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
        lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()

        for _, group in grouped:
            group = group.sort_values('datetime').copy()
            
            # Normalize datetime
            group['datetime_norm'] = group['datetime'].dt.dayofyear / 365.0
            
            # Normalize other features
            group['latitude_norm'] = (group['latitude'] - lat_min) / (lat_max - lat_min)
            group['longitude_norm'] = (group['longitude'] - lon_min) / (lon_max - lon_min)

            # Drop original geo and time columns, keep wildfire for labels
            processed_group = group.drop(columns=['latitude', 'longitude', 'datetime'])

            # Identify numeric features for normalization (excluding already normalized ones and Wildfire)
            numeric_features = processed_group.select_dtypes(include=np.number).columns.tolist()
            numeric_features.remove('Wildfire')
            if 'datetime_norm' in numeric_features: numeric_features.remove('datetime_norm')
            if 'latitude_norm' in numeric_features: numeric_features.remove('latitude_norm')
            if 'longitude_norm' in numeric_features: numeric_features.remove('longitude_norm')

            # Split data
            train_ratio = self.data_config['split_ratios']['train']
            val_ratio = self.data_config['split_ratios']['val']
            
            n = len(processed_group)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_val_set = processed_group.iloc[:val_end]
            
            # Calculate normalization stats from train+val
            stats = train_val_set[numeric_features].agg(['mean', 'std'])
            
            # Normalize numeric features
            for col in numeric_features:
                mean = stats.loc['mean', col]
                std = stats.loc['std', col]
                if std > 0:
                    processed_group[col] = (processed_group[col] - mean) / std
                else:
                    print(f"Column {col} is constant; assigning zeros after normalization.")
                    processed_group[col] = 0 # Constant columns become zero

            if self.split == 'train':
                split_group = processed_group.iloc[:train_end]
            elif self.split == 'val':
                split_group = processed_group.iloc[train_end:val_end]
            else: # test
                split_group = processed_group.iloc[val_end:]

            # Create sequences
            for i in range(len(split_group) - self.window_size):
                sequence_df = split_group.iloc[i:i + self.window_size]
                label_df = split_group.iloc[i:i + self.window_size]
                
                self.sequences.append(torch.tensor(sequence_df.drop('Wildfire', axis=1).values, dtype=torch.float32))
                self.labels.append(torch.tensor(label_df['Wildfire'].values, dtype=torch.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_num_features(self):
        if len(self.sequences) == 0:
            # This is a fallback for an empty dataset.
            # We need to calculate the number of features based on the config.
            required_cols = ["latitude", "longitude", "datetime", "Wildfire"]
            features = self.data_config.get('features', [])

            all_cols = list(set(required_cols + features))
            return len(all_cols) - 1  # Exclude 'Wildfire' label
        return self.sequences[0].shape[1]
