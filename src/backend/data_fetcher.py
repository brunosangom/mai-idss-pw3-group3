from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import requests


class WeatherFetcher(ABC):
    @abstractmethod
    def fetch_data(self, lat, lon, past_days, days) -> dict:
        """Fetch weather data for a given location."""
        pass

class OpenMeteoFetcher(WeatherFetcher):
    def fetch_data(self, lat, lon, past_days, days) -> dict:
        hourly = self._fetch_hourly_data(lat, lon, past_days, days)
        noon_samples = self._preprocess_weather_data(hourly, days)
        return noon_samples


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
            "tas": hourly["temperature_2m"],  # °C
            "hurs": hourly["relative_humidity_2m"],  # %
            "sfcWind": hourly["windspeed_10m"],  # km/h
            "precip": hourly["precipitation"],  # mm
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


import re
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests


class WeatherGovFetcher(WeatherFetcher):
    """
    WeatherFetcher for  https://api.weather.gov
    Rückgabe-DataFrame:
        index: datetime (tz-aware, UTC)
        columns:
            tas      (°C)     -- temperature (ca. 2 m)
            hurs     (%)      -- humidity
            sfcWind  (km/h)   -- wind speed (ca. 10 m)
            precip   (mm/h)   -- precip
            pr       (mm)     -- 24h-precip until noon
    """

    BASE_URL = "https://api.weather.gov"

    _DUR_RE = re.compile(
        r"^P"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?$"
    )

    def __init__(self, user_agent: str = "IDSS-Wildfire/1.0 (example@example.com)"):
        self.user_agent = user_agent
    def fetch_data(self, lat: float, lon: float, past_days: int, days: int) -> pd.DataFrame:
        """
        Fetch weather data in format for xclim
        """
        hourly = self._fetch_hourly_data(lat, lon, past_days, days)
        noon_samples = self._preprocess_weather_data(hourly, days)
        return noon_samples

    def _headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/geo+json",
        }

    def _fetch_point_and_grid_properties(self, lat: float, lon: float):
        """
        fetch:
            - /points/{lat},{lon}
            - forecastGridData-Properties for this gridpoint
        """
        point_url = f"{self.BASE_URL}/points/{lat:.4f},{lon:.4f}"
        r = requests.get(point_url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        point_js = r.json()
        point_props = point_js["properties"]

        grid_url = point_props["forecastGridData"]
        r2 = requests.get(grid_url, headers=self._headers(), timeout=30)
        r2.raise_for_status()
        grid_js = r2.json()
        grid_props = grid_js.get("properties", {})

        return point_props, grid_props

    def _get_nearest_station_id(self, point_props: Dict) -> Optional[str]:
        stations_url = point_props.get("observationStations")
        if not stations_url:
            return None

        r = requests.get(stations_url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        fc = r.json()
        features: List[Dict] = fc.get("features", [])
        if not features:
            return None

        station_url = features[0].get("id")
        if not station_url:
            return None
        return station_url.rstrip("/").split("/")[-1]  # z.B. ".../stations/KLAX" → "KLAX"

    @staticmethod
    def _convert_temp(value: Optional[float], unit_code: Optional[str]) -> float:
        if value is None or np.isnan(value):
            return np.nan
        code = (unit_code or "").split(":")[-1]
        if code.lower() == "degc":
            return float(value)
        if code.lower() == "degf":
            return (float(value) - 32.0) * 5.0 / 9.0
        return float(value)

    @staticmethod
    def _convert_wind(value: Optional[float], unit_code: Optional[str]) -> float:
        """
        convert units to km/h.
        """
        if value is None or np.isnan(value):
            return np.nan
        v = float(value)
        code = (unit_code or "").split(":")[-1]

        if code == "km_h-1":
            return v
        if code in {"m_s-1", "m_s"}:
            return v * 3.6
        if code in {"kn", "kt"}:
            return v * 1.852
        if code in {"mi_h-1"}:
            return v * 1.609344

        return v  # Fallback

    @staticmethod
    def _convert_precip_amount_to_mm(value: Optional[float], unit_code: Optional[str]) -> float:
        if value is None or np.isnan(value):
            return 0.0
        v = float(value)
        code = (unit_code or "").split(":")[-1]

        # 1 kg/m² Wasser ≈ 1 mm
        if code in {"mm", "kg_m-2"}:
            return v
        if code == "in":
            return v * 25.4
        return v

    def _fetch_observations(
        self,
        station_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        url = f"{self.BASE_URL}/stations/{station_id}/observations"
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": 200,
        }
        r = requests.get(url, headers=self._headers(), params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        feats: List[Dict] = js.get("features", [])

        if not feats:
            return pd.DataFrame()

        times = []
        temps = []
        hums = []
        winds = []
        precs = []

        precip_fields = [
            ("precipitationLastHour", 1),
            ("precipitationLast3Hours", 3),
            ("precipitationLast6Hours", 6),
            ("precipitationLast24Hours", 24),
        ]

        for feat in feats:
            props = feat.get("properties", {})
            ts_str = props.get("timestamp")
            if not ts_str:
                continue

            ts = pd.to_datetime(ts_str, utc=True)

            # Temperatur
            t_info = props.get("temperature", {})
            tas = self._convert_temp(t_info.get("value"), t_info.get("unitCode"))

            # relative Feuchte
            rh_info = props.get("relativeHumidity", {})
            rh_val = rh_info.get("value")
            hurs = float(rh_val) if rh_val is not None and not np.isnan(rh_val) else np.nan

            # Wind
            w_info = props.get("windSpeed", {})
            sfc_wind = self._convert_wind(w_info.get("value"), w_info.get("unitCode"))

            # Niederschlag → mm/h (über bevorzugtes Zeitfenster)
            precip_mm_per_hour = 0.0
            for field, hours in precip_fields:
                p_info = props.get(field)
                if not isinstance(p_info, dict):
                    continue
                p_val = p_info.get("value")
                if p_val is None or np.isnan(p_val):
                    continue
                mm_total = self._convert_precip_amount_to_mm(p_val, p_info.get("unitCode"))
                precip_mm_per_hour = mm_total / float(hours)
                break  # das "beste" gefundene Intervall nutzen

            times.append(ts)
            temps.append(tas)
            hums.append(hurs)
            winds.append(sfc_wind)
            precs.append(precip_mm_per_hour)

        if not times:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "tas": temps,
                "hurs": hums,
                "sfcWind": winds,
                "precip": precs,
            },
            index=pd.to_datetime(times, utc=True),
        ).sort_index()

        return df


    @classmethod
    def _parse_duration(cls, dur: str) -> pd.Timedelta:

        m = cls._DUR_RE.match(dur)
        if not m:
            return pd.Timedelta(hours=1)

        days = int(m.group("days") or 0)
        hours = int(m.group("hours") or 0)
        minutes = int(m.group("minutes") or 0)
        seconds = int(m.group("seconds") or 0)
        if days == hours == minutes == seconds == 0:
            hours = 1
        return pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    def _expand_grid_series(self, field: Dict, kind: str) -> pd.Series:
        if not field:
            raise RuntimeError(f"Required field missing for kind='{kind}'")

        uom = field.get("uom") or field.get("unitCode") or ""
        values = field.get("values") or []

        data = {}
        for item in values:
            valid = item.get("validTime")
            value = item.get("value")
            if valid is None or value is None or np.isnan(value):
                continue

            try:
                start_str, dur_str = valid.split("/")
            except ValueError:
                start_str, dur_str = valid, "PT1H"

            # timezone-aware (UTC)
            start = pd.to_datetime(start_str, utc=True)
            delta = self._parse_duration(dur_str)
            hours = int(max(delta.total_seconds() // 3600, 1))

            if kind == "precip":
                per_hour = float(value) / hours
            else:
                per_hour = float(value)

            for h in range(hours):
                t = start + pd.Timedelta(hours=h)
                if kind == "precip":
                    data[t] = data.get(t, 0.0) + per_hour
                else:
                    data[t] = per_hour

        if not data:
            raise RuntimeError(f"No data points found for kind='{kind}'")

        ser = pd.Series(data).sort_index()
        ser = self._convert_series_units(ser, uom, kind)
        return ser

    @staticmethod
    def _convert_series_units(ser: pd.Series, uom: str, kind: str) -> pd.Series:
        if ser.empty:
            return ser

        code = (uom or "").split(":")[-1]  # z.B. "wmoUnit:degC" → "degC"

        if kind == "temp":
            if code.lower().endswith("degc"):
                return ser
            if code.lower().endswith("degf"):
                return (ser - 32.0) * 5.0 / 9.0

        elif kind == "wind":
            if code == "km_h-1":
                return ser
            if code in {"m_s-1", "m_s"}:
                return ser * 3.6
            if code in {"kn", "kt"}:
                return ser * 1.852
            if code in {"mi_h-1"}:
                return ser * 1.609344

        elif kind == "precip":
            if code in {"mm", "kg_m-2"}:
                return ser
            if code == "in":
                return ser * 25.4

        return ser

    def _build_hourly_forecast(self, grid_props: Dict) -> pd.DataFrame:

        temp_field = grid_props.get("temperature")
        rh_field = grid_props.get("relativeHumidity")
        wind_field = grid_props.get("windSpeed")
        pr_field = grid_props.get("quantitativePrecipitation") or grid_props.get("qpf")

        tas = self._expand_grid_series(temp_field, "temp")
        hurs = self._expand_grid_series(rh_field, "rh")
        sfcWind = self._expand_grid_series(wind_field, "wind")
        precip = self._expand_grid_series(pr_field, "precip") if pr_field else None

        df = pd.DataFrame(
            {
                "tas": tas,
                "hurs": hurs,
                "sfcWind": sfcWind,
            }
        )

        if precip is not None:
            df["precip"] = precip
        else:
            df["precip"] = 0.0

        df = df.sort_index()

        for col in ["tas", "hurs", "sfcWind"]:
            df[col] = df[col].interpolate(method="time").ffill().bfill()

        df["precip"] = df["precip"].fillna(0.0)

        return df

    def _fetch_hourly_data(self, lat: float, lon: float, past_days: int, days: int) -> pd.DataFrame:
        now = pd.Timestamp.utcnow().floor("h")

        point_props, grid_props = self._fetch_point_and_grid_properties(lat, lon)

        # -------------------- Observations (Vergangenheit) --------------------
        obs_df = pd.DataFrame()
        if past_days > 0:
            station_id = self._get_nearest_station_id(point_props)
            if station_id:
                # +1 Tag Puffer für 24h-Niederschlag vor erstem Sample
                days_back = max(past_days + 1, 1)
                start_obs = now - pd.Timedelta(days=days_back)
                obs_df = self._fetch_observations(station_id, start_obs, now)
                # ggf. auf [start_obs, now] beschränken (sollte eh so sein)
                obs_df = obs_df[(obs_df.index >= start_obs) & (obs_df.index <= now)]

        # -------------------- Forecast (Zukunft) ------------------------------
        fc_df = self._build_hourly_forecast(grid_props)
        end_forecast = now + pd.Timedelta(days=days + 1)
        fc_df = fc_df[(fc_df.index >= now) & (fc_df.index <= end_forecast)]

        # -------------------- Zusammenführen ---------------------------------
        frames = []
        if not fc_df.empty:
            frames.append(fc_df)
        if not obs_df.empty:
            frames.append(obs_df)

        if not frames:
            raise RuntimeError("Could not load any hourly weather data from weather.gov.")

        hourly = pd.concat(frames).sort_index()

        hourly = hourly[~hourly.index.duplicated(keep="last")]

        for col in ["tas", "hurs", "sfcWind", "precip"]:
            if col not in hourly:
                hourly[col] = 0.0 if col == "precip" else np.nan

        hourly = hourly.sort_index()
        for col in ["tas", "hurs", "sfcWind"]:
            hourly[col] = hourly[col].interpolate(method="time").ffill().bfill()
        hourly["precip"] = hourly["precip"].fillna(0.0)

        return hourly

    @staticmethod
    def _nearest_noon(group: pd.DataFrame) -> pd.Series:
        tz = group.index.tz
        noon = pd.Timestamp(group.index[0].date()).tz_localize(tz) + pd.Timedelta(hours=12)
        diffs_abs = np.abs((group.index - noon).asi8)
        i = int(diffs_abs.argmin())
        return group.iloc[i]

    def _preprocess_weather_data(self, hourly: pd.DataFrame, days: int) -> pd.DataFrame:
        if hourly is None or hourly.empty:
            raise RuntimeError("Could not load hourly fields from weather.gov.")

        df = hourly.sort_index()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        tz = df.index.tz

        unique_dates = sorted({ts.date() for ts in df.index})
        noon_times = []
        noon_rows = []

        for d in unique_dates:
            target_noon = pd.Timestamp(d).tz_localize(tz) + pd.Timedelta(hours=12)
            pos = df.index.get_indexer([target_noon], method="nearest")
            idx = pos[0]
            if idx == -1:
                continue
            noon_times.append(df.index[idx])
            noon_rows.append(df.iloc[idx])

        if not noon_rows:
            raise RuntimeError("No noon samples could be determined from hourly data.")

        noon_samples = pd.DataFrame(noon_rows)
        noon_samples.index = pd.DatetimeIndex(noon_times, tz=tz)

        pr_24h = []
        for tnoon in noon_samples.index:
            window_start = tnoon - pd.Timedelta(hours=24)
            pr_sum = df.loc[window_start:tnoon, "precip"].sum(min_count=1)
            pr_24h.append(float(0.0 if pd.isna(pr_sum) else pr_sum))
        noon_samples["pr"] = pr_24h

        if len(noon_samples) >= 2:
            noon_samples = noon_samples.iloc[1:]

        noon_samples = noon_samples.iloc[:days]
        if noon_samples.empty:
            raise RuntimeError("Too few data points to calculate FWI from weather.gov.")

        if noon_samples.index.tz is not None:
            noon_samples.index = noon_samples.index.tz_convert("UTC").tz_localize(None)

        noon_samples.index.name = "time"

        return noon_samples
