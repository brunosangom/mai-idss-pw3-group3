from xarray.structure.merge import dataset_update_method

from src.backend.data_fetcher import WeatherGovFetcher, OpenMeteoFetcher
from src.backend.model_driven import FWICalcalculator


def test_fwi_calc():
    calculator_gov = FWICalcalculator(fetcher=WeatherGovFetcher())
    calculator_meteo = FWICalcalculator(fetcher=OpenMeteoFetcher())
    vals_gov = calculator_gov.get_fwi( 43.43143875, -89.87114563, days=7, use_season_mask=False)
    vals_meteo = calculator_meteo.get_fwi( 43.43143875, -89.87114563, days=7, use_season_mask=False)
    for i, _ in enumerate(vals_gov):
        print(vals_gov[i])
        print(vals_meteo[i])
        print("#####")

def test_weather_data():
    coord =[47.6062, -122.3321]
    lat = coord[0]
    lon = coord[1]
    past_days = 2
    days = 7

    fetcher_gov = WeatherGovFetcher()
    data_gov = fetcher_gov.fetch_data(lat, lon, past_days, days)

    fetcher_meteo = OpenMeteoFetcher()
    data_meteo = fetcher_meteo.fetch_data(lat, lon, past_days, days)

    print(data_meteo)
    print("###")
    print(data_gov)
test_weather_data()
