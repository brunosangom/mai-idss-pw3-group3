"""
Inference module for wildfire prediction.

This module provides efficient real-time inference capabilities by combining:
1. Data-driven predictions (ML model)
2. Model-driven predictions (Fire Weather Index)

The fire risk assessment combines both approaches following these rules:
- If data-driven predicts No: return FWI prediction ('Very High' -> 'High')
- If data-driven predicts Yes:
  - FWI Low -> 'Moderate'
  - FWI Moderate/High -> 'High'
  - FWI Very High/Extreme -> 'Extreme'
"""

import warnings
import logging
import os
import json
import hashlib
import time as time_module
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import torch

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pint unit redefinition warnings from xclim (used by model_driven module)
# These are harmless - xclim loads custom unit definitions that overlap with pint defaults
logging.getLogger('pint.util').setLevel(logging.ERROR)

from config import ExperimentConfig
from models import create_model
from dataset import WildfireDataset
from data_fetcher import WeatherFetcher


# Fire risk levels
FireRiskLevel = Literal['Low', 'Moderate', 'High', 'Extreme']


class InferenceDataStore:
    """
    Efficient data store for inference that enables fast lookups by (latitude, longitude, date).
    
    Instead of loading the entire preprocessed CSV (which takes 30+ seconds), this class:
    1. Converts CSV to Parquet format for faster loading (~10x speedup)
    2. Creates a location index for O(1) location lookups
    3. Loads only the required window of data for a specific query
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_config = config.get_data_config()
        self.window_size = self.data_config['window_size']
        
        # Cache directory for efficient storage (must be set before _get_feature_cols)
        self.cache_dir = self._get_cache_dir()
        
        # Get feature columns from config
        self.feature_cols = self._get_feature_cols()
        
        # Location index: maps (lat, lon) -> data availability info
        self._location_index: Optional[Dict] = None
        
        # Lazy-loaded dataframe (only load when needed)
        self._df: Optional[pd.DataFrame] = None
        
    def _get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        data_path = Path(self.data_config.get('path', '../../data/'))
        return data_path.parent / "preprocessed_cache"
    
    def _get_cache_key(self) -> str:
        """Generate cache key matching the WildfireDataset cache key."""
        return WildfireDataset._get_cache_key(self.config)
    
    def _get_feature_cols(self) -> list:
        """Load feature columns from cache."""
        cache_key = self._get_cache_key()
        feature_cols_path = self.cache_dir / f"feature_cols_{cache_key}.json"
        
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"Feature columns file not found at {feature_cols_path}. "
                "Please run training first to generate preprocessed data."
            )
    
    def _get_parquet_path(self) -> Path:
        """Get path to the Parquet cache file."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"inference_{cache_key}.parquet"
    
    def _get_index_path(self) -> Path:
        """Get path to the location index file."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"location_index_{cache_key}.json"
    
    def _ensure_parquet_cache(self) -> None:
        """
        Convert CSV cache to Parquet format if not already done.
        Parquet is much faster to load (~10x) and supports efficient filtering.
        """
        parquet_path = self._get_parquet_path()
        
        if parquet_path.exists():
            logger.info(f"Using existing Parquet cache: {parquet_path}")
            return
        
        logger.info("Converting CSV cache to Parquet format (one-time operation)...")
        
        cache_key = self._get_cache_key()
        
        # Load all splits from CSV
        dfs = []
        for split in ['train', 'val', 'test']:
            csv_path = self.cache_dir / f"{split}_{cache_key}.csv"
            if csv_path.exists():
                logger.info(f"Loading {split} split from CSV...")
                df = pd.read_csv(csv_path)
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(
                f"No preprocessed CSV files found in {self.cache_dir}. "
                "Please run training first to generate preprocessed data."
            )
        
        # Combine all splits
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        
        # Sort by location and date for efficient range queries
        combined_df = combined_df.sort_values(['latitude', 'longitude', 'datetime']).reset_index(drop=True)
        
        # Save as Parquet
        combined_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet cache to {parquet_path}")
        
        # Create location index
        self._create_location_index(combined_df)
    
    def _create_location_index(self, df: pd.DataFrame) -> None:
        """
        Create an index mapping (latitude, longitude) to date ranges.
        This enables O(1) lookup to check if a location exists in the dataset.
        """
        index_path = self._get_index_path()
        
        logger.info("Creating location index...")
        
        location_index = {}
        for (lat, lon), group in df.groupby(['latitude', 'longitude']):
            # Round coordinates for consistent lookup
            lat_key = f"{lat:.5f}"
            lon_key = f"{lon:.5f}"
            key = f"{lat_key},{lon_key}"
            
            location_index[key] = {
                'latitude': lat,
                'longitude': lon,
                'min_date': group['datetime'].min().isoformat(),
                'max_date': group['datetime'].max().isoformat(),
                'count': len(group)
            }
        
        # Save index
        with open(index_path, 'w') as f:
            json.dump(location_index, f)
        
        self._location_index = location_index
        logger.info(f"Created location index with {len(location_index)} locations")
    
    def _load_location_index(self) -> Dict:
        """Load location index from file."""
        if self._location_index is not None:
            return self._location_index
        
        index_path = self._get_index_path()
        
        if not index_path.exists():
            # Need to create the index - load data first
            self._ensure_parquet_cache()
            df = pd.read_parquet(self._get_parquet_path())
            df['datetime'] = pd.to_datetime(df['datetime'])
            self._create_location_index(df)
        else:
            with open(index_path, 'r') as f:
                self._location_index = json.load(f)
        
        return self._location_index
    
    def _find_nearest_location(self, latitude: float, longitude: float) -> Optional[Tuple[float, float]]:
        """
        Find the nearest location in the dataset to the given coordinates.
        
        Args:
            latitude: Query latitude
            longitude: Query longitude
            
        Returns:
            Tuple of (nearest_lat, nearest_lon) or None if no location is close enough
        """
        index = self._load_location_index()
        
        # First try exact match (with rounding)
        lat_key = f"{latitude:.5f}"
        lon_key = f"{longitude:.5f}"
        key = f"{lat_key},{lon_key}"
        
        if key in index:
            info = index[key]
            return (info['latitude'], info['longitude'])
        
        # Find nearest location within threshold (e.g., 0.5 degrees)
        threshold = 0.5  # degrees
        min_dist = float('inf')
        nearest = None
        
        for loc_key, info in index.items():
            loc_lat = info['latitude']
            loc_lon = info['longitude']
            
            # Simple Euclidean distance (good enough for small distances)
            dist = np.sqrt((latitude - loc_lat)**2 + (longitude - loc_lon)**2)
            
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                nearest = (loc_lat, loc_lon)
        
        return nearest
    
    def _load_data(self) -> pd.DataFrame:
        """Load the full dataset (lazy loading)."""
        if self._df is not None:
            return self._df
        
        self._ensure_parquet_cache()
        
        logger.info("Loading Parquet data...")
        self._df = pd.read_parquet(self._get_parquet_path())
        self._df['datetime'] = pd.to_datetime(self._df['datetime'])
        logger.info(f"Loaded {len(self._df):,} rows")
        
        return self._df
    
    def get_window(self, latitude: float, longitude: float, date: datetime) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        """
        Get a window of preprocessed features ending at the specified date for the given location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: End date for the window
            
        Returns:
            Tuple of (features_tensor, labels_array) or None if data not available
            - features_tensor: Shape (window_size, num_features)
            - labels_array: Shape (window_size,)
        """
        # Find nearest location
        nearest = self._find_nearest_location(latitude, longitude)
        if nearest is None:
            logger.warning(f"No data available for location ({latitude}, {longitude})")
            return None
        
        actual_lat, actual_lon = nearest
        if (actual_lat, actual_lon) != (latitude, longitude):
            logger.info(f"Using nearest location ({actual_lat}, {actual_lon}) for query ({latitude}, {longitude})")
        
        # Load data
        df = self._load_data()
        
        # Filter to location
        location_mask = (
            (np.abs(df['latitude'] - actual_lat) < 1e-6) & 
            (np.abs(df['longitude'] - actual_lon) < 1e-6)
        )
        location_df = df[location_mask].copy()
        
        if len(location_df) == 0:
            logger.warning(f"No data found for location ({actual_lat}, {actual_lon})")
            return None
        
        # Convert date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        elif not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.to_datetime(date)
        
        # Normalize date to midnight for comparison
        date = pd.Timestamp(date).normalize()
        
        # Filter to window ending at or before the specified date
        location_df = location_df.sort_values('datetime')
        window_start = date - timedelta(days=self.window_size - 1)
        
        window_df = location_df[
            (location_df['datetime'] >= window_start) & 
            (location_df['datetime'] <= date)
        ]
        
        if len(window_df) < self.window_size:
            logger.warning(
                f"Insufficient data for window at ({actual_lat}, {actual_lon}, {date}). "
                f"Need {self.window_size} days, got {len(window_df)}"
            )
            return None
        
        # Take exactly window_size rows (the most recent ones)
        window_df = window_df.tail(self.window_size)
        
        # Extract features and labels
        features = window_df[self.feature_cols].values.astype(np.float32)
        labels = window_df['Wildfire'].values.astype(np.int32)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features)
        
        return features_tensor, labels

    def get_raw_window(self, latitude: float, longitude: float, date: datetime, 
                       window_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Get a window of RAW (un-normalized) weather data for FWI calculation.
        
        This method returns the original weather variables (pr, rmax, rmin, tmmn, tmmx, vs, etc.)
        before normalization, which is needed for FWI calculation.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: End date for the window
            window_days: Number of days in the window (default 7 for FWI)
            
        Returns:
            DataFrame with raw weather data indexed by datetime, or None if not available
        """
        # Find nearest location
        nearest = self._find_nearest_location(latitude, longitude)
        if nearest is None:
            return None
        
        actual_lat, actual_lon = nearest
        
        # Load data
        df = self._load_data()
        
        # Filter to location
        location_mask = (
            (np.abs(df['latitude'] - actual_lat) < 1e-6) & 
            (np.abs(df['longitude'] - actual_lon) < 1e-6)
        )
        location_df = df[location_mask].copy()
        
        if len(location_df) == 0:
            return None
        
        # Convert date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        elif not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.to_datetime(date)
        
        # Normalize date to midnight for comparison
        date = pd.Timestamp(date).normalize()
        
        # Filter to window ending at or before the specified date
        location_df = location_df.sort_values('datetime')
        window_start = date - timedelta(days=window_days - 1)
        
        window_df = location_df[
            (location_df['datetime'] >= window_start) & 
            (location_df['datetime'] <= date)
        ].copy()
        
        if len(window_df) == 0:
            return None
        
        # Set datetime as index for FWI compatibility
        window_df = window_df.set_index('datetime')
        
        return window_df

    def get_ground_truth(self, latitude: float, longitude: float, date: datetime) -> Optional[int]:
        """
        Get the ground truth wildfire label for a given location and date.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date to check
            
        Returns:
            1 if wildfire occurred, 0 if not, None if data not available
        """
        # Find nearest location
        nearest = self._find_nearest_location(latitude, longitude)
        if nearest is None:
            return None
        
        actual_lat, actual_lon = nearest
        
        # Load data
        df = self._load_data()
        
        # Filter to location
        location_mask = (
            (np.abs(df['latitude'] - actual_lat) < 1e-6) & 
            (np.abs(df['longitude'] - actual_lon) < 1e-6)
        )
        location_df = df[location_mask]
        
        # Convert date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        elif not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.to_datetime(date)
        
        date = pd.Timestamp(date).normalize()
        
        # Find the row for this date
        date_mask = location_df['datetime'].dt.normalize() == date
        date_rows = location_df[date_mask]
        
        if len(date_rows) == 0:
            return None
        
        return int(date_rows['Wildfire'].iloc[0])


class WildfirePredictor:
    """
    Data-driven wildfire predictor using a trained ML model.
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_dir: Directory containing model.pt and config.yaml
        """
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.yaml"
        self.model_path = self.model_dir / "model.pt"
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Load config
        self.config = ExperimentConfig(str(self.config_path))
        self.model_config = self.config.get_model_config()
        self.data_config = self.config.get_data_config()
        
        # Get threshold from config
        self.threshold = self.model_config.get('threshold', 0.5)
        
        # Initialize data store
        self.data_store = InferenceDataStore(self.config)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self._model: Optional[torch.nn.Module] = None
    
    def _load_model(self) -> torch.nn.Module:
        """Lazy load the model."""
        if self._model is not None:
            return self._model
        
        logger.info("Loading model...")
        
        # Get number of features from data store
        num_features = len(self.data_store.feature_cols)
        window_size = self.data_config['window_size']
        
        # Create model
        self._model = create_model(
            model_config=self.model_config,
            num_features=num_features,
            window_size=window_size,
            device=self.device
        )
        
        # Load weights
        self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self._model.eval()
        
        logger.info(f"Model loaded: {self.model_config['name']}")
        return self._model
    
    def get_prediction(self, latitude: float, longitude: float, date: datetime) -> Optional[Tuple[bool, float]]:
        """
        Get wildfire prediction for a given location and date.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for prediction
            
        Returns:
            Tuple of (prediction, probability) or None if data not available
            - prediction: True if wildfire predicted, False otherwise
            - probability: Raw model output probability
        """
        # Get data window
        result = self.data_store.get_window(latitude, longitude, date)
        if result is None:
            return None
        
        features, labels = result
        
        # Load model
        model = self._load_model()
        
        # Prepare input: (1, window_size, num_features)
        sequences = features.unsqueeze(0).to(self.device)
        
        # Prepare targets (shifted labels for autoregressive models)
        labels_tensor = torch.from_numpy(labels).unsqueeze(0).to(self.device)
        zero_tensor = torch.zeros(1, 1, dtype=labels_tensor.dtype, device=self.device)
        targets = torch.cat([zero_tensor, labels_tensor[:, :-1]], dim=1)
        
        # Run inference
        with torch.no_grad():
            outputs = model(sequences, targets)  # (1, window_size)
            
            # Get prediction for last timestep
            prob = outputs[0, -1].item()
        
        # Apply threshold
        prediction = prob >= self.threshold
        
        return prediction, prob


class EfficientWeatherFetcher(WeatherFetcher):
    """
    Efficient WeatherFetcher that loads RAW (un-normalized) weather data
    for FWI calculation.
    
    Unlike CsvWeatherFetcher which loads the entire CSV on every instantiation,
    this class:
    1. Creates a Parquet cache of raw data (one-time operation)
    2. Uses the location index from InferenceDataStore for fast lookups
    3. Reuses the already-loaded dataframe across queries
    
    IMPORTANT: This fetcher uses RAW data, not the normalized data from 
    the preprocessed cache, because FWI calculations require actual physical units.
    """
    
    # Class-level cache for raw data (shared across instances)
    _raw_df: Optional[pd.DataFrame] = None
    _raw_parquet_path: Optional[Path] = None
    
    def __init__(self, data_store: InferenceDataStore):
        """
        Initialize with an existing InferenceDataStore.
        
        Args:
            data_store: Pre-initialized InferenceDataStore instance
        """
        self.data_store = data_store
        self._ensure_raw_data_loaded()
    
    def _get_raw_parquet_path(self) -> Path:
        """Get path to the raw data Parquet cache."""
        return self.data_store.cache_dir / "raw_weather_data.parquet"
    
    def _ensure_raw_data_loaded(self) -> None:
        """Load or create the raw weather data cache."""
        parquet_path = self._get_raw_parquet_path()
        
        # Check if already loaded in class cache
        if EfficientWeatherFetcher._raw_df is not None and EfficientWeatherFetcher._raw_parquet_path == parquet_path:
            return
        
        if parquet_path.exists():
            logger.info(f"Loading raw weather data from Parquet cache...")
            EfficientWeatherFetcher._raw_df = pd.read_parquet(parquet_path)
            EfficientWeatherFetcher._raw_df['datetime'] = pd.to_datetime(EfficientWeatherFetcher._raw_df['datetime'])
            EfficientWeatherFetcher._raw_parquet_path = parquet_path
            logger.info(f"Loaded {len(EfficientWeatherFetcher._raw_df):,} rows of raw weather data")
            return
        
        # Need to create the raw data cache from original CSV
        logger.info("Creating raw weather data Parquet cache (one-time operation)...")
        
        csv_path = Path(self.data_store.data_config.get('path', '../../data/Wildfire_Dataset.csv'))
        
        # Load only the columns needed for FWI calculation
        raw_cols = ['latitude', 'longitude', 'datetime', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vs']
        
        logger.info(f"Loading raw data from {csv_path}...")
        raw_df = pd.read_csv(csv_path, usecols=raw_cols)
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
        
        # Apply data quality fixes (same as in dataset.py preprocessing)
        raw_df = self._apply_data_quality_fixes(raw_df)
        
        # Sort by location and date for efficient queries
        raw_df = raw_df.sort_values(['latitude', 'longitude', 'datetime']).reset_index(drop=True)
        
        # Save to Parquet
        raw_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved raw weather data cache to {parquet_path}")
        
        EfficientWeatherFetcher._raw_df = raw_df
        EfficientWeatherFetcher._raw_parquet_path = parquet_path
    
    def _apply_data_quality_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same data quality fixes as dataset.py but keep raw units."""
        # Handle sentinel values (32767 = missing)
        weather_cols = ['pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vs']
        for col in weather_cols:
            if col in df.columns:
                df.loc[df[col] > 10000, col] = np.nan
        
        # Interpolate missing values
        if df.isnull().values.any():
            df = df.interpolate(method='linear', limit_direction='both')
            df = df.ffill().bfill()
        
        # Physical constraints - clipping
        # Humidity (0-100%)
        for col in ['rmax', 'rmin']:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)
        
        # Non-negative variables
        if 'pr' in df.columns:
            df['pr'] = df['pr'].clip(lower=0)
        if 'vs' in df.columns:
            df['vs'] = df['vs'].clip(lower=0)
        
        return df
    
    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points on Earth (in km)."""
        R = 6371.0  # Earth radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def fetch_data(self, lat: float, lon: float, target_date, past_days: int = 4) -> pd.DataFrame:
        """
        Fetch RAW weather data for FWI calculation.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_date: End date for the data window
            past_days: Number of days of data to fetch
            
        Returns:
            DataFrame with columns: tas, hurs, sfcWind, precip, pr
            indexed by datetime (matching CsvWeatherFetcher output format)
        """
        # Find nearest location using the data store's index
        nearest = self.data_store._find_nearest_location(lat, lon)
        if nearest is None:
            raise ValueError(f"No data available for location ({lat}, {lon})")
        
        actual_lat, actual_lon = nearest
        
        # Get raw data
        df = EfficientWeatherFetcher._raw_df
        
        # Filter to location (use looser tolerance for float comparison)
        location_mask = (
            (np.abs(df['latitude'] - actual_lat) < 1e-4) & 
            (np.abs(df['longitude'] - actual_lon) < 1e-4)
        )
        location_df = df[location_mask].copy()
        
        if len(location_df) == 0:
            raise ValueError(f"No raw data found for location ({actual_lat}, {actual_lon})")
        
        # Parse target date
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        target_date = pd.Timestamp(target_date).normalize()
        
        # Filter to window
        window_start = target_date - timedelta(days=past_days - 1)
        location_df = location_df.sort_values('datetime')
        
        window_df = location_df[
            (location_df['datetime'] >= window_start) & 
            (location_df['datetime'] <= target_date)
        ].copy()
        
        if len(window_df) == 0:
            raise ValueError(f"No data available for ({lat}, {lon}) between {window_start} and {target_date}")
        
        # Set datetime as index
        window_df = window_df.set_index('datetime')
        
        # --- Unit conversions (matching CsvWeatherFetcher exactly) ---
        
        # tmmn & tmmx are in Kelvin → mean temp in °C
        tmean_k = (window_df['tmmn'] + window_df['tmmx']) / 2.0
        tas = tmean_k - 273.15
        
        # Relative humidity as mean of daily min & max (already in %)
        hurs = (window_df['rmin'] + window_df['rmax']) / 2.0
        
        # Wind speed vs in m/s → km/h
        sfcWind = window_df['vs'] * 3.6
        
        # Daily precip in mm (already in mm)
        pr_24h = window_df['pr']
        
        # Convert daily total to "at noon" rate in mm/h
        precip_hourly = pr_24h / 24.0
        
        out = pd.DataFrame({
            "tas": tas.values,
            "hurs": hurs.values,
            "sfcWind": sfcWind.values,
            "precip": precip_hourly.values,
            "pr": pr_24h.values,
        }, index=window_df.index)
        
        return out


def get_prediction(latitude: float, longitude: float, date: datetime, 
                   predictor: Optional[WildfirePredictor] = None) -> Optional[Tuple[bool, float]]:
    """
    Get wildfire prediction for a given location and date.
    
    This is a convenience function that creates a predictor if not provided.
    For multiple predictions, create a WildfirePredictor instance once and reuse it.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        date: Date for prediction
        predictor: Optional pre-initialized predictor instance
        
    Returns:
        Tuple of (prediction, probability) or None if data not available
        - prediction: True if wildfire predicted, False otherwise
        - probability: Raw model output probability
    """
    if predictor is None:
        # Get the model directory path relative to this file
        model_dir = Path(__file__).parent / "model"
        predictor = WildfirePredictor(str(model_dir))
    
    return predictor.get_prediction(latitude, longitude, date)


def get_fwi(latitude: float, longitude: float, date: datetime,
            data_store: Optional[InferenceDataStore] = None) -> Optional[Dict[str, Any]]:
    """
    Get Fire Weather Index for a given location and date.
    
    Uses the efficient InferenceDataStore if provided, otherwise falls back
    to the slower CsvWeatherFetcher.
    
    The expected return structure matches FWICalcalculator.get_fwi():
    {
        "date": "2024-01-15",
        "fwi": 12.5,
        "level": "Moderate",  # One of: Low, Moderate, High, Very High, Extreme
        "ffmc": 85.2,
        "dmc": 15.3,
        "dc": 100.5,
        "isi": 5.2,
        "bui": 25.3
    }
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        date: Date for FWI calculation
        data_store: Optional pre-initialized InferenceDataStore for efficient lookup
        
    Returns:
        Dict with FWI values and level, or None if not available
    """
    try:
        from model_driven import FWICalcalculator
        
        # Use efficient fetcher if data_store is provided
        if data_store is not None:
            fetcher = EfficientWeatherFetcher(data_store)
            calculator = FWICalcalculator(fetcher=fetcher)
        else:
            # Fall back to CSV fetcher (slower)
            calculator = FWICalcalculator(fetcher_param="../../data/Wildfire_Dataset.csv")
        
        # Format date as string if needed
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
        
        results = calculator.get_fwi(latitude, longitude, date=date_str, past_days=7)
        if results:
            return results[-1]  # Return the last day (target date)
    except ImportError as e:
        logger.warning(f"FWI module not available: {e}. Using placeholder.")
    except Exception as e:
        logger.warning(f"FWI calculation failed: {e}")
    
    # Placeholder return when FWI is not available
    return {
        "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
        "fwi": None,
        "level": "Moderate",  # Default to moderate when FWI unavailable
        "ffmc": None,
        "dmc": None,
        "dc": None,
        "isi": None,
        "bui": None
    }


def get_fire_risk(latitude: float, longitude: float, date: datetime,
                  predictor: Optional[WildfirePredictor] = None) -> Dict[str, Any]:
    """
    Get combined fire risk assessment from data-driven and model-driven components.
    
    The combination logic:
    - If data-driven predicts No: return FWI prediction ('Very High' -> 'High')
    - If data-driven predicts Yes:
      - FWI Low -> 'Moderate'
      - FWI Moderate/High -> 'High'
      - FWI Very High/Extreme -> 'Extreme'
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        date: Date for risk assessment
        predictor: Optional pre-initialized predictor instance
        
    Returns:
        Dict with fire risk assessment including:
        - risk_level: 'Low', 'Moderate', 'High', or 'Extreme'
        - ml_prediction: bool or None
        - ml_probability: float or None
        - fwi_level: str or None
        - fwi_value: float or None
        - ground_truth: int (0 or 1) or None
        - elapsed_time_ms: float (milliseconds)
    """
    start_time = time_module.perf_counter()
    
    # Get data store from predictor for efficient FWI calculation
    data_store = predictor.data_store if predictor else None
    
    # Get data-driven prediction
    ml_result = get_prediction(latitude, longitude, date, predictor)
    
    # Get model-driven (FWI) prediction using efficient fetcher
    fwi_result = get_fwi(latitude, longitude, date, data_store=data_store)
    
    # Get ground truth
    ground_truth = None
    if data_store:
        ground_truth = data_store.get_ground_truth(latitude, longitude, date)
    
    # Default FWI level if not available
    fwi_level = fwi_result.get('level', 'Moderate') if fwi_result else 'Moderate'
    fwi_value = fwi_result.get('fwi') if fwi_result else None
    
    # Map 'Very High' to 'High' for output (we only have 4 levels)
    def normalize_fwi_level(level: str) -> FireRiskLevel:
        if level == 'Very High':
            return 'High'
        elif level in ('Low', 'Moderate', 'High', 'Extreme'):
            return level
        else:
            return 'Moderate'  # Default for unknown levels like "Out of fire season"
    
    # Determine risk level
    ml_prediction = None
    ml_probability = None
    
    if ml_result is None:
        logger.warning("ML prediction not available, using FWI only")
        risk_level = normalize_fwi_level(fwi_level)
    else:
        ml_prediction, ml_probability = ml_result
        
        # Apply combination logic
        if not ml_prediction:
            # Data-driven predicts No wildfire
            risk_level = normalize_fwi_level(fwi_level)
        else:
            # Data-driven predicts Yes wildfire
            if fwi_level == 'Low':
                risk_level = 'Moderate'
            elif fwi_level in ('Moderate', 'High'):
                risk_level = 'High'
            elif fwi_level in ('Very High', 'Extreme'):
                risk_level = 'Extreme'
            else:
                # Default for unknown FWI levels
                risk_level = 'High'
    
    elapsed_time = (time_module.perf_counter() - start_time) * 1000  # Convert to ms
    
    return {
        'risk_level': risk_level,
        'ml_prediction': ml_prediction,
        'ml_probability': ml_probability,
        'fwi_level': fwi_level,
        'fwi_value': fwi_value,
        'ground_truth': ground_truth,
        'elapsed_time_ms': round(elapsed_time, 2)
    }


# Global predictor instance for reuse
_predictor: Optional[WildfirePredictor] = None


def initialize_predictor(model_dir: str = "model") -> WildfirePredictor:
    """
    Initialize the global predictor instance.
    
    Call this once at application startup for efficient repeated predictions.
    
    Args:
        model_dir: Directory containing model.pt and config.yaml
        
    Returns:
        Initialized WildfirePredictor instance
    """
    global _predictor
    _predictor = WildfirePredictor(model_dir)
    return _predictor


def get_predictor() -> Optional[WildfirePredictor]:
    """Get the global predictor instance."""
    return _predictor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wildfire Risk Prediction")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--model-dir", type=str, default="model", help="Model directory")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = initialize_predictor(args.model_dir)
    
    # Parse date
    date = datetime.strptime(args.date, "%Y-%m-%d")
    
    # Get fire risk (returns full result dict now)
    result = get_fire_risk(args.lat, args.lon, date, predictor)
    
    print(f"\n{'='*60}")
    print(f"Fire Risk Assessment for ({args.lat}, {args.lon}) on {args.date}")
    print(f"{'='*60}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Elapsed Time: {result['elapsed_time_ms']:.2f} ms")
    print(f"\nComponent Predictions:")
    if result['ml_prediction'] is not None:
        print(f"  ML Prediction: {'Yes' if result['ml_prediction'] else 'No'} (probability: {result['ml_probability']:.3f})")
    else:
        print(f"  ML Prediction: Not available")
    print(f"  FWI Level: {result['fwi_level']} (FWI: {result['fwi_value']})")
    print(f"\nGround Truth:")
    if result['ground_truth'] is not None:
        gt_str = "Yes (Wildfire)" if result['ground_truth'] == 1 else "No (No Wildfire)"
        print(f"  Actual: {gt_str}")
        # Check if prediction was correct
        if result['ml_prediction'] is not None:
            correct = result['ml_prediction'] == (result['ground_truth'] == 1)
            print(f"  ML Prediction Correct: {'✓' if correct else '✗'}")
    else:
        print(f"  Actual: Not available")
    print(f"{'='*60}")
