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

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Literal
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig
from models import create_model
from dataset import WildfireDataset
from model_driven import FWICalcalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def get_fwi(latitude: float, longitude: float, date: datetime) -> Optional[Dict[str, Any]]:
    """
    Get Fire Weather Index for a given location and date.
    
    This function should be implemented in model_driven. For now, it returns a placeholder.
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
        
    Returns:
        Dict with FWI values and level, or None if not available
    """
    try:        
        calculator = FWICalcalculator()
        results = calculator.get_fwi(latitude, longitude, days=1)
        if results:
            return results[0]
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
                  predictor: Optional[WildfirePredictor] = None) -> FireRiskLevel:
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
        Fire risk level: 'Low', 'Moderate', 'High', or 'Extreme'
    """
    # Get data-driven prediction
    ml_result = get_prediction(latitude, longitude, date, predictor)
    
    # Get model-driven (FWI) prediction
    fwi_result = get_fwi(latitude, longitude, date)
    
    # Default FWI level if not available
    fwi_level = fwi_result.get('level', 'Moderate') if fwi_result else 'Moderate'
    
    # Map 'Very High' to 'High' for output (we only have 4 levels)
    def normalize_fwi_level(level: str) -> FireRiskLevel:
        if level == 'Very High':
            return 'High'
        elif level in ('Low', 'Moderate', 'High', 'Extreme'):
            return level
        else:
            return 'Moderate'  # Default for unknown levels like "Out of fire season"
    
    # If ML prediction is not available, fall back to FWI only
    if ml_result is None:
        logger.warning("ML prediction not available, using FWI only")
        return normalize_fwi_level(fwi_level)
    
    ml_prediction, ml_probability = ml_result
    
    # Apply combination logic
    if not ml_prediction:
        # Data-driven predicts No wildfire
        return normalize_fwi_level(fwi_level)
    else:
        # Data-driven predicts Yes wildfire
        if fwi_level == 'Low':
            return 'Moderate'
        elif fwi_level in ('Moderate', 'High'):
            return 'High'
        elif fwi_level in ('Very High', 'Extreme'):
            return 'Extreme'
        else:
            # Default for unknown FWI levels
            return 'High'


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
    
    # Get fire risk
    risk = get_fire_risk(args.lat, args.lon, date, predictor)
    
    print(f"\nFire Risk Assessment for ({args.lat}, {args.lon}) on {args.date}:")
    print(f"  Risk Level: {risk}")
    
    # Also show component predictions
    ml_result = get_prediction(args.lat, args.lon, date, predictor)
    if ml_result:
        pred, prob = ml_result
        print(f"  ML Prediction: {'Yes' if pred else 'No'} (probability: {prob:.3f})")
    
    fwi_result = get_fwi(args.lat, args.lon, date)
    if fwi_result:
        print(f"  FWI Level: {fwi_result.get('level')} (FWI: {fwi_result.get('fwi')})")
