@echo off
echo Starting Wildfire Prediction IDSS - Development Mode
echo ====================================================
echo.

echo Starting Flask Backend (Port 5000)...
start "Flask Backend" cmd /k "cd src\backend && flask --app app run --debug"

timeout /t 3 /nobreak > nul

echo Starting React Frontend (Port 3000)...
start "React Frontend" cmd /k "cd src\frontend && npm start"

echo.
echo ====================================================
echo Both services are starting in separate windows:
echo - Backend: http://localhost:5000
echo - Frontend: http://localhost:3000
echo ====================================================
echo.
echo Press any key to close this window (services will continue running)
pause > nul
