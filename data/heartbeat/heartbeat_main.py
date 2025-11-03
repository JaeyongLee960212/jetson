# main.py
import multiprocessing
from multiprocessing.spawn import freeze_support
from heartbeat_server import UDP_Server
from heartbeat_config import db_account
from log import get_logger
import pymysql
#test
import csv
import datetime

import keyboard
import os
import psutil
import time

class DB_Handler:
    def __init__(self):
        self.logger = get_logger("DB_LOG")
        self.db_conn, self.curs = self.connect_DB()

    def __del__(self):
        try:
            self.curs.close()
            self.db_conn.close()
        except:
            pass

    def connect_DB(self):
        try:
            conn = pymysql.connect(
                host=db_account['host'],
                user=db_account['user'],
                passwd=db_account['passwd'],
                db=db_account['db'],
                charset=db_account['charset'],
                cursorclass=pymysql.cursors.DictCursor
            )
            curs = conn.cursor()
            self.logger.info("Connect Successfully")
            return conn, curs
        except Exception as e:
            self.logger.error(f"Connect Failed: {e}")
            raise
        #test
    #def insert_heartbeat_tb(self, message):
    #    try:
    #        with open("heartbeat_log.csv", "a", newline="") as f:
    #            writer = csv.writer(f)
    #            writer.writerow([
    #                message['vehicleID'],
    #                message['agencyId'],
    #                message['sensorId'],
    #                message['timeStamp'],
    #                message['heartBeat'],
    #                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #            ])
    #        self.logger.info(f"CSV Logged: {message}")
    #    except Exception as e:
    #        self.logger.error(f"CSV Write Failed: {e}")

    def insert_heartbeat_tb(self, message):
        try:
            sql = """
                INSERT INTO heartbeat_tb (vehicleId, agencyId, sensorId, timeStamp, heartBeat)
                VALUES (%s, %s, %s, %s, %s)
            """
            data = (
                message['vehicleID'],
                message['agencyId'],
                message['sensorId'],
                message['timeStamp'],
                message['heartBeat']
            )
            self.curs.execute(sql, data)
            self.db_conn.commit()
            self.logger.info(f"Insert Success(HeartBeat): {data}")
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Insert Failed(HeartBeat): {e}")

def start_udp_server(ip, port, label):
    server = UDP_Server(ip, port, label)
    server.run()

def exit_program():
    process = psutil.Process(os.getpid())
    for proc in process.children(recursive=True):
        print(f"child process {proc.pid} terminate")
        proc.kill()
    print(f"Parent process {process.pid} terminate")
    process.kill()

if __name__ == "__main__":
    freeze_support()
    #포트가 같아야 함 
    server_configs = [
        ("192.168.10.110", 4041, "NUVO"),
        ("192.168.10.111", 4041, "ODYSSEY"),
        ("192.168.10.119", 4041, "LINE"),
        ("192.168.10.118", 4041, "JETSON"),
    ]

    processes = []
    for ip, port, label in server_configs:
        p = multiprocessing.Process(target=start_udp_server, args=(ip, port, label))
        p.start()
        print(f"[MAIN] Started process for {label}")
        processes.append(p)

    print("@@@@@@@@@@@@@@@@@@@@   종료 : Q or ESC   @@@@@@@@@@@@@@@@@@@")
    keyboard.add_hotkey('q', exit_program)
    keyboard.add_hotkey('esc', exit_program)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit_program()

    os.system('pause')
