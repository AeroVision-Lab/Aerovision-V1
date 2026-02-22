#!/bin/bash
# Linux/Mac 快捷启动脚本
cd "$(dirname "$0")"
python3 run_workflow.py "$@"
