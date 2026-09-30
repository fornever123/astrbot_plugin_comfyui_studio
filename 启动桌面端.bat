@echo off
chcp 65001 >nul
cd /d "%~dp0"
python desktop\desktop_app.py
if errorlevel 1 pause
