# Model-driven approach
* Calculate risk of Fire using the FWI (Forest Fire Weather Index) system.
* 
* Advantages
  * integrations multiple weather parameters in a single measure of fire potential
  * widely used and accepted in wildfire management worldwide [1]
  * data availability from standard meteorological observations
  * Easier to implement compared to more complex models
* Disadvantes
  * In the US FWI is less used than NFDRS (National Fire Danger Rating System) -> may limit adoption to US
  * Simplified Assumptions
  * Daily resolution
  * Does not take in account the ignition likelihood 
## Calculation
* Input data
  * Air Temperature (°C)
  * Relative Humidity (%)
  * Wind Speed (km/hr)
  * Precipitation (mm/day)
* Intermediate Components (estimated based on input data, especially precipitation)
  * Fine Fuel Moisture Code (FFMC): moisture content of litter and other cured fine fuels
  * Duff Moisture Code (DMC): moisture content of loosely compacted organic layers of moderate depth
  * Drought Code (DC): moisture content of deep, compact organic layers
* Fire behavior indices
  * Initial Spread Index (ISI): rate of fire spread
  * Build Up Index (BUI): amount of fuel available for combustion

# Implementation
* Calculation done using xclim
* Data from open-meteo
  * Usage Limit: ⚠️ 600 calls / min	, 5000 calls/hour, 10.00 calls/day

  ## Usage Example
  * Get the prediction for the next 7 days for coordinate 39.7392/-104.9903
  ```python
    calculator = FWICalcalculator()
    vals = calculator.get_fwi(39.7392, -104.9903, days=7) 
```

# Appendix
[1] NOTE: NOT PEER REVIEWD- find better source for report :) Janine A. Baijnath-Rodino, Efi Foufoula-Georgiou, Tirtha Banerjee. Reviewing the “Hottest” Fire Indices Worldwide. ESS Open Archive . July 31, 2020. 