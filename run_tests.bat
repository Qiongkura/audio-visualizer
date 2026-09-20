@echo off
rem ---------------------------------------------------------------
rem  Run the unit tests (no sound card required).
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

set PY=venv\Scripts\python.exe
if not exist %PY% set PY=python

%PY% -m unittest discover -s tests -t . -v
pause
