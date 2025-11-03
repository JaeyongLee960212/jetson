@echo off
echo Activating conda base environment...
call C:\Users\KETI\miniconda3\Scripts\activate.bat base


echo Start heartbeat program
python heartbeat_main.py

pause
