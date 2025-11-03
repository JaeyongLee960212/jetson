# udp_heartbeat_client.py
import socket
import time
from datetime import datetime
import heartbeat_config as config
import json


class UDP_Client:
    def __init__(self, server_ip, server_port, label):
        self.server_addr = (server_ip, server_port)
        self.label = label
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def get_timestamp(self):
        try:
            now = datetime.now()
            time_string = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            time_string = time_string[:-7] + time_string[-7:-3]
            #year = _now.year
            #month = int(f"{_now.month:02d}")      
            #day = _now.day
            #hour = _now.hour
            #minute = _now.minute
            #second = _now.second
            #millisecond = int(_now.microsecond / 1000)
            #second_millisecond = int(f"{second:02d}{millisecond:03d}")
            return time_string
        
        except Exception as e:
            self.logger.error(f"DB_ERROR {e}")
            return None

    def send_heartbeat(self):
        # Sensor ID by label
        label_to_sensor = {
            "NUVO": "ET_INFRA_MARWIS_CPU",
            "ODYSSEY": "ET_INFRA_TUNNELLUX_CPU",
            "LINE": "ET_INFRA_SURFACEPOTHOLE_CPU",
            "JETSON": "ET_INFRA_VIDEOPROCESSING_JETSON"
        }
        sensorID = label_to_sensor.get(self.label)
        if not sensorID:
            print("@@@@@@@@@@@@@@@@@@@@@Error: Invalid Label@@@@@@@@@@@@@@@@@@@@@@")
            return
        
        time_string = self.get_timestamp()
        try:
            # ONSTART
            time_string = self.get_timestamp()
            message = {
                "vehicle_id": config.Vehicle,
                "agency_id": config.Agency,
                "sensor_id": sensorID,
                "heart_beat": "ONSTART",                
                #"year": year,
                #"month": month,
                #"day": day,
                #"hour": hour,
                #"minute": minute,
                #"millisecond": second_millisecond,
                "time_stamp": time_string,
            }
            self.socket.sendto(json.dumps(message).encode(), self.server_addr)
            print(f"[{self.label}] HEARTBEAT: {message}")

            # ON
            while True:
                time_string = self.get_timestamp()
                #message["year"] = year
                #message["month"] = month
                #message["day"] = day
                #message["hour"] = hour
                #message["minute"] = minute
                #message["millisecond"] = second_millisecond
                message["time_stamp"] = time_string
                message["heart_beat"] = "ON"
                self.socket.sendto(json.dumps(message).encode(), self.server_addr)
                print(f"[{self.label}] HEARTBEAT: {message}")
                time.sleep(60)

        except KeyboardInterrupt:
            # Ctrl+c
            time_string = self.get_timestamp()
            #message["year"] = year
            #message["month"] = month
            #message["day"] = day
            #message["hour"] = hour
            #message["minute"] = minute
            #message["millisecond"] = second_millisecond
            message["time_stamp"] = time_string
            message["heart_beat"] = "OFF"
            self.socket.sendto(json.dumps(message).encode(), self.server_addr)
            print(f"[{self.label}] HEARTBEAT: {message}")

        except:
            # exception
            time_string = self.get_timestamp()
            message = {
                "vehicle_id": config.Vehicle,
                "agency_id": config.Agency,
                "sensor_id": sensorID,
                "heart_beat": "OFF",
                #"year": year,
                #"month": month,
                #"day": day,
                #"hour": hour,
                #"minute": minute,
                #"millisecond": second_millisecond,
                "time_stamp" : time_string
            }
            self.socket.sendto(json.dumps(message).encode(), self.server_addr)
            print(f"[{self.label}] HEARTBEAT : {message}")
            print(f"@@@@@@@@@@@@@Error: Exception@@@@@@@@@@@@@@@@@@@@@@@")


if __name__ == "__main__":
    client = UDP_Client("192.168.10.110", 4041, "JETSON")
    print("Waiting for Server...")
    time.sleep(10)
    client.send_heartbeat()


