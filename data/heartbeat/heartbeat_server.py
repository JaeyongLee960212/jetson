# heartbeat_server.py
import socket
import json
from heartbeat_db_handler import DB_Handler  

class UDP_Server:
    def __init__(self, ip, port, label="Server"):
        self.label = label
        self.db_handler = DB_Handler() 
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((ip, port))
        if ip == "192.168.10.110":
            client_label = "NUVO"
        elif ip == "192.168.10.111":
            client_label = "ODYSSEY"
        elif ip == "192.168.10.118":
            client_label = "JETSON"
        elif ip == "192.168.10.119":
            client_label = "LINE"
        print(f"[{self.label}] Listening on {client_label}{ip}:{port}")

    def run(self):
        while True:
            try:
                data, addr = self.socket.recvfrom(65535)
                message = json.loads(data.decode())
                if message['sensor_id'] == "ET_INFRA_MARWIS_CPU":
                    client_label = "NUVO"
                if message['sensor_id'] == "ET_INFRA_TUNNELLUX_CPU":
                    client_label = "ODYSSEY"
                if message['sensor_id'] == "ET_INFRA_VIDEOPROCESSING_JETSON":
                    client_label = "JETSON"
                if message['sensor_id'] == "ET_INFRA_SURFACEPOTHOLE_CPU":
                    client_label = "LINE"
                print(f"[{client_label}] {client_label}:{addr}:{message}")

                self.db_handler.insert_heartbeat_tb(message)
                print("DB insert complete.")

            except Exception as e:
                print(f"@@@@ ERROR: {e} @@@@")
