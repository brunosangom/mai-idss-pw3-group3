import numpy as np
import requests
import pandas as pd
import xarray as xr
from typing import List, Dict

import xclim

class FWICalcalculator:
    def __init__(self):
        # precipitation threshold for FWI calculations
        self.prec_thresh = "1.5 mm/d"

    def get_fwi(self, lat: float,
                lon: float,
                days: int = 7,
                past_days: int = 2,
                *,
                overwintering: bool = True,
                dc_start: float = 15,
                dmc_start: float = 6,
                ffmc_start: float = 85,
                use_season_mask = True) -> List[Dict]:
        """
        Calculate daily FWI values for a given location using Open-Meteo data and xclim.
        """
        if days < 2:
            days = 2

        hourly = self._fetch_hourly_data(lat, lon, past_days, days)
        noon_samples = self._preprocess_weather_data(hourly, days)

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
            overwintering=overwintering,
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

    @staticmethod
    def _nearest_noon(group: pd.DataFrame) -> pd.Series:
        """
        Take data point approximately at noon (12:00) – robust for pandas 2.3.3 (no .abs() on TimedeltaIndex).
        """
        tz = group.index.tz
        noon = pd.Timestamp(group.index[0].date()).tz_localize(tz) + pd.Timedelta(hours=12)
        diffs_abs = np.abs((group.index - noon).asi8)
        i = int(diffs_abs.argmin())
        return group.iloc[i]

    @staticmethod
    def _fetch_hourly_data(lat, lon, past_days, days):
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat:.5f}&longitude={lon:.5f}"
            "&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
            "&timezone=auto"
            f"&past_days={past_days}"
            f"&forecast_days={max(days, 7)}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        js = resp.json()
        hourly = js.get("hourly", {})
        return hourly

    def _preprocess_weather_data(self, hourly, days):
        if not hourly or "time" not in hourly:
            raise RuntimeError("Could not load hourly fields.")

        df = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "tas": hourly["temperature_2m"],   # °C
            "hurs": hourly["relative_humidity_2m"],  # %
            "sfcWind": hourly["windspeed_10m"],      # km/h
            "precip": hourly["precipitation"],       # mm
        }).set_index("time").sort_index()

        # Find noon sample per day
        noon_samples = df.groupby(df.index.date).apply(self._nearest_noon)
        noon_samples.index = pd.to_datetime(noon_samples.index.astype(str))

        # 24h precipitation up to noon: (t-24h, t]
        pr_24h = []
        for tnoon in noon_samples.index:
            window_start = tnoon - pd.Timedelta(hours=24)
            pr_sum = df.loc[window_start:tnoon, "precip"].sum(min_count=1)
            pr_24h.append(float(0.0 if pd.isna(pr_sum) else pr_sum))
        noon_samples["pr"] = pr_24h

        # Drop the first day for clean 24h windows
        if len(noon_samples) >= 2:
            noon_samples = noon_samples.iloc[1:]

        noon_samples = noon_samples.iloc[:days]
        if noon_samples.empty:
            raise RuntimeError("Too few data points to calculate FWI.")
        return noon_samples


# --- Example call ---
if __name__ == "__main__":
    calculator = FWICalcalculator()
    vals = calculator.get_fwi(39.7392, -104.9903, days=7)
    for row in vals:
        print(row)
