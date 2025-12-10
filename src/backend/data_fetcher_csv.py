import math
import warnings
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from src.backend.data_fetcher import WeatherFetcher

class CsvWeatherFetcher(WeatherFetcher):
    """
    WeatherFetcher implementation for a wildfire-style CSV with columns:
    latitude,longitude,datetime,Wildfire,pr,rmax,rmin,sph,srad,tmmn,tmmx,vs,bi,
    fm100,fm1000,erc,etr,pet,vpd

    For a given (lat, lon, target_date, past_days), this class:
    - searches ONLY stations that have data in that date window,
    - chooses the closest such station,
    - warns if that station is farther than warn_distance_km.
    """

    def __init__(
        self,
        csv_path: str,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        datetime_col: str = "datetime",
        warn_distance_km: float = 50.0,
    ) -> None:
        self.csv_path = csv_path
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.datetime_col = datetime_col
        self.warn_distance_km = warn_distance_km

        self._df = self._load_csv()

        # Pre-group by station to speed up date-window checks
        self._station_groups = self._build_station_groups()

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        return df

    def _build_station_groups(self):
        """
        Builds a dict:
            (lat, lon) -> DataFrame indexed by datetime
        """
        station_groups = {}
        grouped = self._df.groupby([self.lat_col, self.lon_col])
        for (slat, slon), grp in grouped:
            grp = grp.sort_values(self.datetime_col).copy()
            grp = grp.set_index(self.datetime_col)
            station_groups[(float(slat), float(slon))] = grp
        return station_groups

    # --- distance helpers -------------------------------------------------

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        """
        Great-circle distance between two points on Earth (in km).
        Inputs in degrees.
        """
        R = 6371.0  # Earth radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
            dlambda / 2
        ) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _normalize_date(self, target_date) -> date:
        if isinstance(target_date, (datetime, pd.Timestamp)):
            return target_date.date()
        if isinstance(target_date, date):
            return target_date
        # allow string like "2018-08-15"
        return pd.to_datetime(target_date).date()

    def _select_station_for_window(self, lat, lon, start_day, target_day):
        """
        Among all stations that have at least one row between start_day and target_day
        (by calendar date), pick the closest to (lat, lon).

        Returns
        -------
        station_lat, station_lon, dist_km, group_df

        Raises
        ------
        ValueError if no station has data in that window.
        """
        best_key = None
        best_dist = float("inf")

        for (slat, slon), grp in self._station_groups.items():
            idx_dates = grp.index.date
            mask = (idx_dates >= start_day) & (idx_dates <= target_day)
            if not np.any(mask):
                continue  # this station has no data in the requested window

            dist = self._haversine_km(lat, lon, slat, slon)
            if dist < best_dist:
                best_dist = dist
                best_key = (slat, slon)

        if best_key is None:
            raise ValueError(
                f"No station has data between {start_day} and {target_day}."
            )

        slat, slon = best_key
        grp = self._station_groups[best_key]

        if best_dist > self.warn_distance_km:
            warnings.warn(
                f"Requested location ({lat:.4f}, {lon:.4f}) is "
                f"{best_dist:.1f} km away from nearest station with data "
                f"in the requested window ({slat:.4f}, {slon:.4f}).",
                RuntimeWarning,
            )

        # Filter group to the exact window (we already know it has at least one row)
        idx_dates = grp.index.date
        mask = (idx_dates >= start_day) & (idx_dates <= target_day)
        window = grp.loc[mask].copy()

        return slat, slon, best_dist, window

    def fetch_data(self, lat, lon, target_date, past_days=4) -> pd.DataFrame:
        """
        Return multi-day window needed for FWI, ending on target_date (inclusive).

        Example:
        --------
        target_date = "2018-08-15", past_days = 7
        -> rows where date is between 2018-08-09 and 2018-08-15 (if present).
        """
        if past_days is None or past_days <= 0:
            raise ValueError("past_days must be a positive integer")

        target_day = self._normalize_date(target_date)
        start_day = target_day - timedelta(days=past_days - 1)

        # Pick closest station that actually has data in that window
        _, _, _, window = self._select_station_for_window(
            lat, lon, start_day, target_day
        )

        if window.empty:
            # Shouldn't happen, we already checked in _select_station_for_window
            raise ValueError(
                f"No CSV entries found between {start_day} and {target_day} "
                f"for any station."
            )
        # --- Unit conversions / feature mapping (vectorized) ---

        # tmmn & tmmx assumed Kelvin → mean temp in °C
        tmean_k = (window["tmmn"] + window["tmmx"]) / 2.0
        tas = tmean_k - 273.15

        # Relative humidity as mean of daily min & max
        hurs = (window["rmin"] + window["rmax"]) / 2.0

        # Wind speed vs in m/s → km/h
        sfcWind = window["vs"] * 3.6

        # Daily precip in mm
        pr_24h = window["pr"]

        # Convert daily total to "at noon" rate in mm/h
        precip_hourly = pr_24h / 24.0

        out = pd.DataFrame(
            {
                "tas": tas,
                "hurs": hurs,
                "sfcWind": sfcWind,
                "precip": precip_hourly,
                "pr": pr_24h,
            },
            index=window.index,
        )
        return out
    def _find_nearest_station(self, lat: float, lon: float):
        """
        Find nearest (station_lat, station_lon, distance_km) from cached stations.
        """
        if self._stations.size == 0:
            raise ValueError("No stations available in CSV.")

        # vectorized haversine
        lats = self._stations[:, 0]
        lons = self._stations[:, 1]

        # compute distances
        # (loop is fine unless you have millions of stations, then you might KD-tree it)
        distances = [
            self._haversine_km(lat, lon, slat, slon)
            for slat, slon in zip(lats, lons)
        ]
        distances = np.array(distances)
        idx_min = int(distances.argmin())

        nearest_lat = float(lats[idx_min])
        nearest_lon = float(lons[idx_min])
        dist_km = float(distances[idx_min])

        return nearest_lat, nearest_lon, dist_km


    def debug_station_coverage(self, lat, lon):
        # Find nearest station
        station_lat, station_lon, dist_km = self._find_nearest_station(lat, lon)
        print(f"Nearest station to ({lat}, {lon}) is "
              f"({station_lat}, {station_lon}) at {dist_km:.1f} km")

        # All rows for that station
        df = self._df
        mask = (
                (df[self.lat_col] == station_lat) &
                (df[self.lon_col] == station_lon)
        )
        subset = df.loc[mask].copy()
        subset[self.datetime_col] = pd.to_datetime(subset[self.datetime_col], utc=True)

        subset = subset.sort_values(self.datetime_col)
        subset = subset.set_index(self.datetime_col)

        print("Date range for this station:")
        print("  min:", subset.index.min())
        print("  max:", subset.index.max())

        print("\nUnique dates (first 20):")
        print(sorted({d.date() for d in subset.index})[:20])

        return subset


if __name__ == "__main__":
    vals = calculator.get_fwi(39.7392, -104.9903, "2018-08-15")
    for row in vals:
        print(row)
