@echo off
rem One-click 1.5B run: same steps as run_pilot.bat with configs/laptop-1.5b.yaml.
rem Everything is written to run_1.5b.log in this folder.
setlocal
cd /d "%~dp0"
set LOG=%~dp0run_1.5b.log
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo === 1.5b run started %DATE% %TIME% === > "%LOG%"

set CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" set CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat
if not exist "%CONDA_BAT%" (
  echo RUN FAILED: could not find conda under %USERPROFILE% >> "%LOG%"
  goto :end
)

call "%CONDA_BAT%" env list | findstr /b /c:"kwik " >nul
if errorlevel 1 (
  echo --- creating conda env kwik --- >> "%LOG%"
  call "%CONDA_BAT%" create -n kwik python=3.11 -y >> "%LOG%" 2>&1
)
call "%CONDA_BAT%" activate kwik >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: could not activate env kwik >> "%LOG%"
  goto :end
)

echo --- installing --- >> "%LOG%"
python -m pip install -e ".[serve,dev]" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: pip install >> "%LOG%"
  goto :end
)

echo --- tests --- >> "%LOG%"
python -m pytest -q >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: tests >> "%LOG%"
  goto :end
)

echo --- 1.5b run --- >> "%LOG%"
echo start %TIME% >> "%LOG%"
kwik all --config configs/laptop-1.5b.yaml >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: kwik all >> "%LOG%"
  goto :end
)
echo end %TIME% >> "%LOG%"
echo RUN DONE >> "%LOG%"

:end
echo === finished %DATE% %TIME% === >> "%LOG%"
endlocal
