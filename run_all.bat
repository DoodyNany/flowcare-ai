@echo off
REM Flow Care AI - rebuild everything on your own Windows PC (no Claude, no internet needed after install)
cd /d "%~dp0"
echo [1/6] Installing Python packages (first time only)...
python -m pip install -r requirements.txt || goto :err
echo [2/6] Simulating the hospital...
python sim\simulate_hospital.py || goto :err
echo [3/6] Training bed forecast + week-ahead planner...
python models\train_forecast.py || goto :err
echo [4/6] Training patient risk AI...
python models\train_risk.py || goto :err
if exist xray_data\pneumoniamnist_128.npz (
  echo [5/6] Training chest X-ray AI - takes about 15-25 minutes...
  python models\train_xray.py 128 || goto :err
) else (
  echo [5/6] Skipped X-ray training: put pneumoniamnist_128.npz in xray_data. Using the saved model.
)
if exist xray_data\nodulemnist3d.npz (
  echo [5/6] Training CT nodule AI - takes about 30 minutes...
  python models\train_ct.py || goto :err
) else (
  echo [5/6] Skipped CT training: put nodulemnist3d.npz in xray_data. Using the saved model.
)
echo [6/6] Building the dashboard...
python web\build.py || goto :err
start "" web\flowcare_ai.html
echo Done. The dashboard opened in your browser.
pause
exit /b 0
:err
echo Something failed above. Read the red error message, or check that Python 3.11+ is installed.
pause
exit /b 1
