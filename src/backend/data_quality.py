import logging
import os
from typing import List

import numpy as np
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)


class DataQualityInspector:
    """
    Validates FireCastRL dataset integrity.
    Checks for physical constraints, missing values, and unit mismatches.
    """

    # Valid physical ranges (SI units / Standard Meteorological bounds)
    CONSTRAINTS = {
        # --- API Variables (Live Data) ---
        "hurs": (0, 100),      # Humidity (%)
        "tas": (-60, 60),      # Temperature (°C)
        "precip": (0, 1000),   # Precipitation (mm)
        "sfcWind": (0, 300),   # Wind Speed (km/h)

        # --- Dataset Variables (FireCastRL) ---
        "pr": (0, 1000),       # Precip (mm/day)
        "rmax": (0, 100),      # Max Humidity (%)
        "rmin": (0, 100),      # Min Humidity (%)
        "sph": (0, 0.05),      # Specific Humidity (kg/kg)
        "srad": (0, 1361),     # Solar Radiation (W/m^2)
        "tmmn": (-60, 60),     # Min Temp (°C)
        "tmmx": (-60, 60),     # Max Temp (°C)
        "vs": (0, 100),        # Wind Speed (m/s)
        "vpd": (0, 10),        # Vapor Pressure Deficit (kPa)
        "fm100": (0, 100),     # Fuel Moisture (%)
        "fm1000": (0, 100),    # Fuel Moisture (%)
        "erc": (0, 200),       # Energy Release Component
        "bi": (0, 400),        # Burning Index
        "etr": (0, 20),        # Evapotranspiration
        "pet": (0, 20),        # Potential Evapotranspiration
    }

    def __init__(self, df: pd.DataFrame, context: str = "Raw Data"):
        self.df = df
        self.context = context
        self.report_lines: List[str] = []
        self._issues_found = False

    def run(self) -> str:
        self._log(f"=== QUALITY REPORT: {self.context.upper()} ===")
        self._check_shape()
        self._check_missing()
        self._check_physics()
        self._log(f"=== END {self.context.upper()} ===\n")
        return "\n".join(self.report_lines)

    def _log(self, msg: str):
        self.report_lines.append(msg)

    def _check_shape(self):
        rows, cols = self.df.shape
        self._log(f"[1] Structure: {rows} rows, {cols} columns")
        
        dupes = self.df.duplicated().sum()
        if dupes > 0:
            self._issues_found = True
            self._log(f"    ! WARNING: Found {dupes} duplicate rows")

    def _check_missing(self):
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        
        if missing.empty:
            self._log("[2] Missing Data: None")
        else:
            self._issues_found = True
            self._log(f"[2] Missing Data: Found issues in {len(missing)} columns")
            for col, count in missing.items():
                pct = (count / len(self.df)) * 100
                self._log(f"    - {col}: {count} missing ({pct:.2f}%)")

    def _check_physics(self):
        self._log("[3] Physical Consistency Check")
        clean_pass = True
        
        for col, (min_v, max_v) in self.CONSTRAINTS.items():
            if col not in self.df.columns:
                continue
                
            mask_out = (self.df[col] < min_v) | (self.df[col] > max_v)
            count = mask_out.sum()
            
            if count > 0:
                clean_pass = False
                self._issues_found = True
                
                # Diagnostic for Temperature Units
                if col in ['tmmn', 'tmmx'] and count > 0.5 * len(self.df):
                    mean_val = self.df.loc[mask_out, col].mean()
                    self._log(f"    ! CRITICAL: {col} has {count} failures. Mean outlier value: {mean_val:.1f}.")
                    if mean_val > 200:
                        self._log(f"      -> DIAGNOSIS: Data is likely in KELVIN. Expected CELSIUS.")
                else:
                    self._log(f"    ! WARNING: {col} has {count} values outside [{min_v}, {max_v}]")

        if clean_pass:
            self._log("    - All checks passed.")


# ---------------------------------------------------------
# Helper function to mimic dataset.py cleaning logic
# ---------------------------------------------------------
def apply_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies strict cleaning for FireCastRL dataset.
    1. Deduplicates
    2. Removes Sentinels (32767) -> NaN
    3. Converts Kelvin -> Celsius
    4. Imputes & Clips
    """
    df = df.copy()
    
    # 1. Deduplicate
    df = df.drop_duplicates()
    
    # 2. Sentinel Removal (Crucial step missing in previous check!)
    numeric_cols = df.select_dtypes(include=np.number).columns
    weather_cols = [c for c in numeric_cols if c not in ['latitude', 'longitude', 'year']]
    
    for col in weather_cols:
        # Values > 30000 are sentinels (missing data)
        df.loc[df[col] > 30000, col] = np.nan

    # 3. Unit Conversion (Kelvin to Celsius)
    temp_cols = ['tmmn', 'tmmx']
    for col in temp_cols:
        if col in df.columns:
            # Heuristic: If median is > 100, it's definitely Kelvin
            if df[col].median() > 100:
                df[col] = df[col] - 273.15

    # 4. Imputation (Time-series interpolation)
    if df.isnull().values.any():
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(method='bfill').fillna(method='ffill')

    # 5. Safety Clipping
    # Humidity caps
    for col in ['rmax', 'rmin', 'fm100', 'fm1000']:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)
    
    # Non-negative constraints
    pos_cols = ['pr', 'sph', 'srad', 'vs', 'vpd', 'bi', 'erc', 'etr', 'pet']
    for col in pos_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            
    # Temperature safety clip
    for col in temp_cols:
        if col in df.columns:
            df[col] = df[col].clip(-60, 60)

    return df


if __name__ == "__main__":
    # Path configuration
    DATA_PATH = "../../data/Wildfire_Dataset.csv"
    OUTPUT_REPORT = "../../docs/DATA_QUALITY_REPORT.txt"
    
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading {DATA_PATH}...")
        df_raw = pd.read_csv(DATA_PATH)
        
        # Run inspection on RAW
        inspector_raw = DataQualityInspector(df_raw, context="Raw Data (Before)")
        report_raw = inspector_raw.run()
        
        # Apply fixes
        logger.info("Applying cleaning rules...")
        df_clean = apply_cleaning_rules(df_raw)
        
        # Run inspection on CLEAN
        inspector_clean = DataQualityInspector(df_clean, context="Processed Data (After)")
        report_clean = inspector_clean.run()
        
        # Save full report
        final_report = report_raw + "\n" + ("-" * 40) + "\n" + report_clean
        
        os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print("\n" + final_report)
        logger.info(f"Report saved to {OUTPUT_REPORT}")
    else:
        logger.error(f"File not found: {DATA_PATH}")