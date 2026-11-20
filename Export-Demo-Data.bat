@echo off
rem Downloads the Sentinel-2 data used for the demo video into the demo_data folder.
cd /d "%~dp0backend"
".venv\Scripts\python.exe" -m app.demo_export
pause
