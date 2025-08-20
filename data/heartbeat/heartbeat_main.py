# heartbeat_main.py
import multiprocessing
from multiprocessing.spawn import freeze_support
from heartbeat_server import UDP_Server
import keyboard
import os
import psutil
import time

def start_udp_server(ip, port, label):
    server = UDP_Server(ip, port, label)
    server.run()

def exit_program():
    process = psutil.Process(os.getpid())
    for proc in process.children(recursive=True):
        print(f"Child process {proc.pid} terminated")
        proc.kill()
    print(f"Main process {process.pid} terminated")
    process.kill()

if __name__ == "__main__":
    freeze_support()

    server_configs = [
        ("192.168.10.110", 4041, "NUVO"),
        ("192.168.10.111", 4041, "ODYSSEY"),
        ("192.168.10.118", 4041, "JETSON"),
        ("192.168.10.119", 4041, "LINE"),
    ]

    processes = []
    for ip, port, label in server_configs:
        p = multiprocessing.Process(target=start_udp_server, args=(ip, port, label))
        p.start()
        print(f"[MAIN] Started process for {label}")
        processes.append(p)

    print("@@@@@@@@@@ 종료 : Q 또는 ESC @@@@@@@@@@")
    keyboard.add_hotkey('q', exit_program)
    keyboard.add_hotkey('esc', exit_program)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit_program()
