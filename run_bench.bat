@echo off
rem One-click benchmark of the confidence API on the 1.5B run (about 5-10 minutes).
rem Everything is written to run_bench.log in this folder.
setlocal
cd /d "%~dp0"
set LOG=%~dp0run_bench.log
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo === bench started %DATE% %TIME% === > "%LOG%"

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

echo --- bench --- >> "%LOG%"
echo start %TIME% >> "%LOG%"
kwik bench --config configs/laptop-1.5b.yaml --n 50 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo RUN FAILED: kwik all >> "%LOG%"
  goto :end
)
echo end %TIME% >> "%LOG%"
echo RUN DONE >> "%LOG%"

:end
echo === finished %DATE% %TIME% === >> "%LOG%"
endlocal
