# Jetson Docker System

## Models
1. **VMS**  
   : Based on YOLOv8s, Trained by 60 images, Class : 2 (VMS_ON, VMS_OFF)
2. **LCS**  
   : Based on YOLOv8s, Trained by 17,394 images, Class : 2 (LCS_ROAD_USUABLE, LCS_ROAD_UNUSUABLE, LCS_BROKEN)
3. **Traffic Light**  
   : Based on YOLOv8s, Trained by 740,724 images, Class : 6 (TRAFFIC_LIGHT_GREEN etc...)
4. **Traffic Sign**  
   : Based on YOLOv8s, Trained by 489,429 images, Class : 82 (TRAFFIC_SIGN_YIELD etc...)

## Structure
<pre>
📂 Jetson-containers/data
 ├── 📁 heartbeat
 │   ├── 📄 heartbeat_client.py
 │   ├── 📄 heartbeat_config.py
 │   ├── 📄 heartbeat_db_handler.py
 │   ├── 📄 heartbeat_jetson_client.sh (For Jetson Only)
 │   ├── 📄 heartbeat_server.bat (For Nuvo Only)
 │   ├── 📄 heartbeat_main.py
 │   ├── 📄 heartbeat_client.bat (For Odyssey, Nuvo, Line)
 │   ├── 📄 heartbeat_server.py
 │   └── 📄 log.py
 ├── 📁 YOLOv8_VMS(VMS Folder)(Editing....)
 │   ├── 📄 VMS_best.pt
 │   ├── 📄 run_track_v8_VMS.py (Main Python File for VMS Model)
 │   ├── 📄 run_v8_VMS.sh (Executable File For Docker Compose)
 │   ├── 📁 Conf
 │   │   └── 📄 config.py (Config File such as Main Server IP, Cam Size, FTP Info, Class ID etc...)
 │   ├── 📁 results (안 쓰는 폴더, 곧 삭제될 예정)
 │   ├── 📁 test 
 │   │   ├── 📄 py_force_ip.py(When you cannot access to the IP Camera, run this file)
 │   │   ├── 📄 py_ipconfig_manual.py
 │   │   └── 📄 py_live_stream.py
 │   ├── 📁 trackers
 │   │   ├── 📄 botsort.yaml (We use this Tracking Method for now)
 │   │   ├── 📄 bytetrack.yaml
 │   │   └── 📄 bytetrack-bak.yaml  
 │   └── 📁 utils
 │   │   ├── 📄 check_vms.py (Settings for Check ROI Options, Threshold and Size Parameters)
 │   │   ├── 📄 comm_utils_vms.py (Settings for FTP, UDP, JSON Format)
 │   │   ├── 📄 lcd_capture.py (Settings for Multicast, Pixel Format, Cam IP etc...)
 │   │   ├── 📄 loader.py (No update since 2023)
 │   │   └── 📄 profiler.py (No update since 2023)
 ├── 📁 YOLOv8_LCS(LCS Folder)
 │   ├── 📄 LCS_best.pt
 │   ├── 📄 run_track_v8_LCS.py (Main Python File for LCS Model)
 │   ├── 📄 run_v8_LCS.sh (Executable File For Docker Compose)
 │   ├── 📁 Conf
 │   │   └── 📄 config.py (Config File such as Main Server IP, Cam Size, FTP Info, Class ID etc...)
 │   ├── 📁 results (안 쓰는 폴더, 곧 삭제될 예정)
 │   ├── 📁 test 
 │   │   ├── 📄 py_force_ip.py(When you cannot access to the IP Camera, run this file)
 │   │   ├── 📄 py_ipconfig_manual.py
 │   │   └── 📄 py_live_stream.py
 │   ├── 📁 trackers
 │   │   ├── 📄 botsort.yaml (We use this Tracking Method for now)
 │   │   ├── 📄 bytetrack.yaml
 │   │   └── 📄 bytetrack-bak.yaml  
 │   └── 📁 utils
 │   │   ├── 📄 check_lcs.py (Settings for Check ROI Options, Threshold and Size Parameters)
 │   │   ├── 📄 comm_utils_lcs.py (Settings for FTP, UDP, JSON Format)
 │   │   ├── 📄 lcd_capture.py (Settings for Multicast, Pixel Format, Cam IP etc...)
 │   │   ├── 📄 loader.py (No update since 2023)
 │   │   └── 📄 profiler.py (No update since 2023)
 ├── 📁 YOLOv8_TS(TS Folder)(Editing....)
 │   ├── 📄 TS_best.pt
 │   ├── 📄 run_track_v8_TS.py (Main Python File for VMS Model)
 │   ├── 📄 run_v8_TS.sh (Executable File For Docker Compose)
 │   ├── 📁 Conf
 │   │   └── 📄 config.py (Config File such as Main Server IP, Cam Size, FTP Info, Class ID etc...)
 │   ├── 📁 results (안 쓰는 폴더, 곧 삭제될 예정)
 │   ├── 📁 test 
 │   │   ├── 📄 py_force_ip.py(When you cannot access to the IP Camera, run this file)
 │   │   ├── 📄 py_ipconfig_manual.py
 │   │   └── 📄 py_live_stream.py
 │   ├── 📁 trackers
 │   │   ├── 📄 botsort.yaml (We use this Tracking Method for now)
 │   │   ├── 📄 bytetrack.yaml
 │   │   └── 📄 bytetrack-bak.yaml  
 │   └── 📁 utils
 │   │   ├── 📄 check_ts.py (Settings for Check ROI Options, Threshold and Size Parameters)
 │   │   ├── 📄 comm_utils_ts.py (Settings for FTP, UDP, JSON Format)
 │   │   ├── 📄 lcd_capture.py (Settings for Multicast, Pixel Format, Cam IP etc...)
 │   │   ├── 📄 loader.py (No update since 2023)
 │   │   └── 📄 profiler.py (No update since 2023)
 ├── 📁 YOLOv8_TL(TL Folder)(Editing....)
 │   ├── 📄 TL_best.pt
 │   ├── 📄 run_track_v8_TL.py (Main Python File for VMS Model)
 │   ├── 📄 run_v8_TL.sh (Executable File For Docker Compose)
 │   ├── 📁 Conf
 │   │   └── 📄 config.py (Config File such as Main Server IP, Cam Size, FTP Info, Class ID etc...)
 │   ├── 📁 results (안 쓰는 폴더, 곧 삭제될 예정)
 │   ├── 📁 test 
 │   │   ├── 📄 py_force_ip.py(When you cannot access to the IP Camera, run this file)
 │   │   ├── 📄 py_ipconfig_manual.py
 │   │   └── 📄 py_live_stream.py
 │   ├── 📁 trackers
 │   │   ├── 📄 botsort.yaml (We use this Tracking Method for now)
 │   │   ├── 📄 bytetrack.yaml
 │   │   └── 📄 bytetrack-bak.yaml  
 │   └── 📁 utils
 │   │   ├── 📄 check_tl.py (Settings for Check ROI Options, Threshold and Size Parameters)
 │   │   ├── 📄 comm_utils_tl.py (Settings for FTP, UDP, JSON Format)
 │   │   ├── 📄 lcd_capture.py (Settings for Multicast, Pixel Format, Cam IP etc...)
 │   │   ├── 📄 loader.py (No update since 2023)
 │   │   └── 📄 profiler.py (No update since 2023)
 ├── 📁 test(Image Save Folder for Debugging)
 │   └── 🖼️ Detected Image
 ├── 📄 ArenaSDK_v0.1.78_LinuxARM64.tar.gz (Before Running arena_api~.whl, You should have to unzip this first and run file "/data/ArenaSDK_LinuxARM64/Arena_SDK_ARM64.conf")
 ├── 📄 arena_api.2.7.1-py3-none-any.whl (Main Install File for Arena API(IP Camera))
 └── 📄 docker-compose.yaml (Main File for Running Docker Compose)
</pre>

### Getting Started
<pre>
   1. git --version 
   2. 사용하고자 하는 폴더 내로 들어가서 git init
   3. git config --global -user.name 'JaeyongLee960212'
   4. git config --global user.password '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' (토큰)
   5. git config --global credential.helper store
   6. git remote add origin https://github.com/JaeyongLee960212/jetson.git
   7. git fetch
   8. git branch -r
   9. git checkout main or jetson
   10. git pull
   11. git add .
   12. git commit -m "새로운 내용 설명"
   13. git push
   <When error occurs because of Large Files>
      1. nano .gitignore
         <Copy&Paste>
         data/**
         
         !data/YOLOv8_*/
         !data/YOLOv8_*/**
         
         data/YOLOv8_*/*.mp4
         data/YOLOv8_*/*.avi
         data/YOLOv8_*/*.pt
         
         !data/YOLOv8_*/**/*.py
         !data/YOLOv8_*/**/*.sh
         !data/YOLOv8_*/**/*.yaml
         !data/*.yml
      2. Ctrl+O and then Press Enter
      3. Ctrl+X to get out
      4. Finish
</pre>

### History
Last updated on 2025-08-13 13:22:00
