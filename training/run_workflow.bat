@echo off
REM Windows 快捷启动脚本
cd /d "%~dp0"
python run_workflow.py %*
