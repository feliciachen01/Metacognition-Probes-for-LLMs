@echo off
rem One-click pilot: creates the conda env (first time only), installs, tests, runs the pilot.
rem Everything is written to run_pilot.log in this folder.
setlocal
cd /d "%~dp0"
set LOG=%~dp0run_pilot.log
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo === pilot started %DATE% %TIME% === > "%LOG%"

set CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" set CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" (
  echo PILOT FAILED: could not find conda under %USERPROFILE% >> "%LOG%"
  goto :end
)

call "%CONDA_BAT%" env list | findstr /b /c:"kwik " >nul
if errorlevel 1 (
  echo --- creating conda env kwik --- >> "%LOG%"
  call "%CONDA_BAT%" create -n kwik python=3.11 -y >> "%LOG%" 2>&1
)
call "%CONDA_BAT%" activate kwik >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PILOT FAILED: could not activate env kwik >> "%LOG%"
  goto :end
)

echo --- installing --- >> "%LOG%"
python -m pip install -e ".[serve,dev]" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PILOT FAILED: pip install >> "%LOG%"
  goto :end
)

echo --- tests --- >> "%LOG%"
python -m pytest -q >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PILOT FAILED: tests >> "%LOG%"
  goto :end
)

echo --- pilot run --- >> "%LOG%"
echo start %TIME% >> "%LOG%"
kwik all --config configs/pilot.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PILOT FAILED: kwik all >> "%LOG%"
  goto :end
)
echo end %TIME% >> "%LOG%"
echo PILOT DONE >> "%LOG%"

:end
echo === finished %DATE% %TIME% === >> "%LOG%"
endlocal
