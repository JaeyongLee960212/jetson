#add function for Socket Comm to Main Server
import cv2
import json
import socket
import ftplib

from datetime import datetime
from io import BytesIO
from conf.config import MAIN_SERVER_FTP_ID, MAIN_SERVER_FTP_PASSWD, FTP_SAVE_PATH

class Comm:
    def __init__(self, main_server_ip, main_server_udp_port) -> None:

        self.main_server_ip = main_server_ip
        self.main_server_udp_port = main_server_udp_port

        #udp socks
        self.udp_sock = self.udp_connect()
        #tcp sock (not use)
        #self.tcp_sock = self.tcp_connect()

    #######################
    ## UDP realated func ##
    #######################
            
    def udp_connect(self) -> None: # add. Connect UDP
        self.sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("UDP Connect")

    def send_data_to_udp(self, msg) -> None: # add. Message Send(UDP)
        self.sock_udp.sendto(msg.encode(), (self.main_server_ip , self.main_server_udp_port))
        print("udp send ", msg)


    #######################
    ## TCP realated func ##
    #######################

    # add . Connect TCP
    def tcp_connect(self)-> None: # add. Connect TCP
        self.sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("wait connect main server....")

        self.sock_tcp.connect((self.main_server_ip, self.main_server_tcp_port))
        print("TCP stream connect!")
        
    # add . Frame Send (TCP)
    def send_frame_to_tcp(self, frame):
        d = frame.flatten()
        s = d.tobytes()
        self.sock_tcp.sendall(s)

    #######################
    ## FTP realated func ##
    #######################
        
    # add. create directory in nuvo ftp server
    def create_dir_ftp(self, ftp, dir):
        parts = dir.split('/')
        for part in parts:
            try:
                ftp.cwd(part)
            except:
                ftp.mkd(part)
                ftp.cwd(part)

    #add. upload frame_capture img to keti ftp server
    def upload_to_ftp(self, image, cls_name ,filename):
        try:
            save_path = f"{FTP_SAVE_PATH}/"
            capture_img = BytesIO()
            success = cv2.imencode('.jpg', image)[1].tobytes()
            capture_img.write(success)    
            capture_img.seek(0)

            with ftplib.FTP(self.main_server_ip, MAIN_SERVER_FTP_ID, MAIN_SERVER_FTP_PASSWD) as ftp:
                self.create_dir_ftp(ftp, save_path)
                print(f"{save_path+filename}")
                ftp.storbinary(f'STOR {save_path+filename}', capture_img)

        except Exception as e:
            print(e)


    #######################
    ##     Com utils     ##
    #######################
    # add. Detected Data Convert to JSON
    def pack_data(self, cls, cls_name, conf, xywh, fc, detect_time, img_file_name):

        save_path = f"{FTP_SAVE_PATH}{cls_name}/"

        bbox_x = xywh[0]
        bbox_y = xywh[1]
        bbox_w = xywh[2]
        bbox_h = xywh[3]
        
        if cls == 2:
            status = 'OFF'
        else:
            status = 'ON'
    
        js_data ={
            'type' : '11',
            'cls_name' : str(cls_name),
            'cls' : cls,
            'conf' : conf,
            'x' : bbox_x,
            'y' : bbox_y,
            'w' : bbox_w,
            'h' : bbox_h,
            'frame_count' : fc,
            'detect_img_path' : save_path + img_file_name,
            'detect_status' : status,
            'detect_time' : detect_time
        }

        json_data = json.dumps(js_data, ensure_ascii=False, indent = '\t')
        return json_data

