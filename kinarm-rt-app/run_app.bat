@echo off
REM Start the KINARM RT app on Windows. Opens in your web browser.
cd /d "%~dp0"

where streamlit >nul 2>nul
if %ERRORLEVEL%==0 (
  echo Starting the app... a browser tab should open. Close this window to stop it.
  streamlit run app.py
  goto :eof
)

echo Streamlit was not found on your PATH.
echo.
echo Set it up once, then re-run this file:
echo   Option A ^(conda, recommended^):
echo     conda env create -f environment.yml ^&^& conda activate kinarm-rt
echo   Option B ^(pip^):
echo     python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
echo.
pause
