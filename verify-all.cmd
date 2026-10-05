@echo off
REM IRIS - run every verification suite.
cd /d "%~dp0"
backend\.venv\Scripts\python.exe scripts\run_all_tests.py %*
