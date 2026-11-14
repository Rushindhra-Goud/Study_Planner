@echo off
REM Double-click or run in the project folder. Creates the venv once, installs, checks, then makes a plan.
cd /d "%~dp0"
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
python -m pip install -q -r requirements.txt
python -m study_planner ingest
python -m study_planner check
python -m study_planner plan --weeks 4 --hours 6
pause
