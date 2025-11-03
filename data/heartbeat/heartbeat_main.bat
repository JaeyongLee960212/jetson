@echo off
echo Activating conda base environment...
call conda activate base

echo Start heartbeat program
python heartbeat_main.py

pause
