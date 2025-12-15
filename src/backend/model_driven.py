import numpy as np
import requests
import pandas as pd
import xarray as xr
from typing import List, Dict
import warnings

# Suppress pint unit redefinition warnings from xclim
# These are harmless - xclim loads custom unit definitions that overlap with pint defaults
warnings.filterwarnings('ignore', message='Redefining.*', module='pint.util')

import xclim

from data_fetcher import WeatherFetcher, OpenMeteoFetcher, WeatherGovFetcher
from data_fetcher_csv import CsvWeatherFetcher


class FWICalcalculator:
    """Calculate the Fire Weather Index (FWI) using Open-Meteo data and xclim."""
    def __init__(self, fetcher: WeatherFetcher = None, fetcher_param : str = None):
        # precipitation threshold for FWI calculations
        self.prec_thresh = "1.5 mm/d"
        if fetcher is None and fetcher_param is not None:
            self.fetcher = CsvWeatherFetcher(fetcher_param)
        else:
            self.fetcher = fetcher


    def get_fwi(self, lat: float,
                lon: float,
                date: str = "2023-01-01",
                past_days: int = 2,
                *,
                overwintering: bool = True,
                dc_start: float = 15,
                dmc_start: float = 6,
                ffmc_start: float = 85,
                use_season_mask = False) -> List[Dict]:
        """
        Calculate daily FWI values for a given location using Open-Meteo data and xclim.
        """

        noon_samples = self.fetcher.fetch_data(lat, lon, date, past_days)
        
        return self.calculate_fwi_series(
            noon_samples, lat, 
            overwintering=overwintering,
            dc_start=dc_start,
            dmc_start=dmc_start,
            ffmc_start=ffmc_start,
            use_season_mask=use_season_mask
        )

    def calculate_fwi_series(self, noon_samples: pd.DataFrame, lat: float,
                             overwintering: bool = True,
                             dc_start: float = 15,
                             dmc_start: float = 6,
                             ffmc_start: float = 85,
                             use_season_mask = False) -> List[Dict]:
        """
        Calculate FWI series from a DataFrame of noon weather samples.
        """
        # --- Build xarray Dataset with explicit UNITS (FIX) ---
        # ensure numeric dtypes (avoids object dtype sneaking in)
        ns = noon_samples.astype(float).copy()
        ds = xr.Dataset(
            data_vars=dict(
                tas=("time", ns["tas"].values, {"units": "degC"}),        # °C (noon)
                pr=("time", ns["pr"].values, {"units": "mm/d"}),          # 24h precip to noon (mm)
                hurs=("time", ns["hurs"].values, {"units": "%"}),         # % (noon)
                sfcWind=("time", ns["sfcWind"].values, {"units": "km/h"}),# km/h (noon, 10 m)
            ),
            coords=dict(
                time=("time", ns.index),
                # FIX: give latitude proper units so xclim's unit checks pass
                lat=([], float(lat), {"units": "degrees_north"}),
            ),
        )

        # Optional: Fire-season mask; set unit to "1" (dimensionless) for clean checks (FIX)
        if use_season_mask:
            season_mask = xclim.indices.fire.fire_season(
                tas=ds.tas, method="WF93", freq="YS",
                temp_start_thresh="12 degC", temp_end_thresh="5 degC",
                temp_condition_days=3
            ).assign_attrs(units="1")
        else:
            season_mask = None

        # --- Compute FWI components (bereits vorhanden) ---
        out = xclim.indices.fire.cffwis_indices(
            tas=ds.tas,
            pr=ds.pr,
            hurs=ds.hurs,
            sfcWind=ds.sfcWind,
            lat=ds.lat,
            overwintering=use_season_mask,
            dc_start=dc_start,
            dmc_start=dmc_start,
            ffmc_start=ffmc_start,
            prec_thresh=self.prec_thresh,
            season_mask=season_mask,
        )

        # --- Access results (FIX) ---
        # cffwis_indices returns   NamedTuple mit Attributen "DC","DMC","FFMC","ISI","BUI","FWI".
        dc_da = out.DC
        dmc_da = out.DMC
        ffmc_da = out.FFMC
        isi_da = out.ISI
        bui_da = out.BUI
        fwi_da = out.FWI

        # Convert o pandas
        fwi = fwi_da.to_series()
        ffmc = ffmc_da.to_series()
        dmc = dmc_da.to_series()
        dc = dc_da.to_series()
        isi = isi_da.to_series()
        bui = bui_da.to_series()

        # Create Results
        result = []
        for t in fwi.index:
            val = float(fwi.loc[t])
            result.append({
                "date": pd.to_datetime(t).date().isoformat(),
                "fwi": round(val, 2),
                "level": self._fwi_level(val),
                "ffmc": round(float(ffmc.loc[t]), 2),
                "dmc": round(float(dmc.loc[t]), 2),
                "dc": round(float(dc.loc[t]), 2),
                "isi": round(float(isi.loc[t]), 2),
                "bui": round(float(bui.loc[t]), 2),
            })
        return result

    @staticmethod
    def _fwi_level(fwi: float) -> str:
        """Simple classification of FWI levels."""
        if pd.isna(fwi):
            return "Out of fire season"
        elif fwi < 5:
            return "Low"
        elif fwi < 12:
            return "Moderate"
        elif fwi < 21:
            return "High"
        elif fwi < 32:
            return "Very High"
        else:
            return "Extreme"




# --- Example call ---
if __name__ == "__main__":
    calculator = FWICalcalculator(fetcher_param="../../data/Wildfire_Dataset.csv")
    vals = calculator.get_fwi(39.7392, -104.9903, "2020-03-01")
    for row in vals:
        print(row)
