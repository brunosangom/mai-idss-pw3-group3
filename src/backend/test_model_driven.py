""" Note: API key does not work for some reason :("""
import requests
import my_secrets
# --- Konfiguration ---
API_KEY = my_secrets.api_key  # Deinen OpenWeather API-Key eintragen
print(f"Using API Key {API_KEY}")
lat, lon = 39.7392, -104.9903  # Beispiel: Denver, Colorado

# --- Anfrage ---
url = f"https://api.openweathermap.org/data/2.5/fwi?lat={lat}&lon={lon}&appid={API_KEY}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    fwi_value = data.get("fwi", None)
    danger = data.get("danger_rating", None)

    print(f"📍 Koordinaten: ({lat}, {lon})")
    print(f"🔥 Fire Weather Index (FWI): {fwi_value}")
    print(f"🚨 Gefahrenstufe: {danger}")
else:
    print("Fehler:", response.status_code, response.text)
