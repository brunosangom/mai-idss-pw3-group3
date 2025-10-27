# Model-driven approach
* Caluclate Risk of wildfire manually using the McArthur Forest Fire Danger Index (FFDI) [1]
* Required data : Keetch-Byram drought index, Formula, Temperature, humidity, ..., number of days since last rain

## Calculation
FFDI = 1.25 * D * exp [ (T - H)/30.0 + 0.0234 * V]

Where:
- D = drought factor,
- T = Temperature (ºC),
- H = humidity (%), and
- V = wind speed (km hr-1).

D = (0.191 * ( I + 104) * (N + 1)^ 1.5) / (3.52 * (N+1)^1.5 + P -1)

- P = precipitation (mm day-1),
- N = number of days since last rain, and
- I is based on Keetch-Byram drought index.


[1] https://catalogue.ceda.ac.uk/uuid/1d9929b28e79491585373e69337cee65/

* 
