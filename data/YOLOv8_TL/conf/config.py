from datetime import datetime
TODAY = detect_time = datetime.now().strftime("%Y_%m_%d")

# integrate server ip (Nuvo)
MAIN_SERVER_IP = "192.168.10.110"
MAIN_SERVER_SUBNETMASK = "255.255.255.0"
MAIN_SERVER_GATEWAY = "192.168.10.254"


# nuvo ftp server 
MAIN_SERVER_FTP_ID = "keti"
MAIN_SERVER_FTP_PASSWD = "keti1234"
FTP_SAVE_PATH = "/front_cam/"+TODAY+"/"

# integrate server port (udp, tcp)
MAIN_SERVER_UDP_PORT = 3939
MAIN_SERVER_TCP_PORT = 3838

# front cam 
LCD_CAMERA_IP = "192.168.10.116"

CAM_WIDTH = 1280
CAM_HEIGHT = 720

#CAM_WIDTH = 1920
#CAM_HEIGHT = 1080

#CAM_WIDTH = 640
#CAM_HEIGHT = 480

# set fps
FPS = 10

#AUG_V8_WEIGHTS_DIR_V1_2 = "/home/keti/ssd/v8_weights/custom_v8s_v1.2/tl/"
AUG_V8_WEIGHTS_DIR_V1_2 = "/home/traffic/tl_only/traffic_light/test2"

# V8_WEIGHTS_DIR_V1_0 = "/home/keti/ssd/v8_weights/custom_v8s_v1.0/"
# V8_WEIGHTS_DIR_V1_1 = "/home/keti/ssd/v8_weights/custom_v8s_v1.1/"
# V8_WEIGHTS_DIR_YOLO = "/home/keti/ssd/v8_weights/yolov8/"

#TEST_VID_DIR = "/home/keti/ssd/test_video/2024_04_29/"
TEST_VID_DIR = "/home/traffic/test_video/"

LIGHT_CLS_DIR = {
    0: "LCS_ROAD_USABLE",
    1: "LCS_ROAD_UNUSABLE",
    2: "TRAFFICSIGNAL_BROKEN",
    3: "TRAFFICSIGNAL_RED",
    4: "TRAFFICSIGNAL_YELLOW",
    5: "TRAFFICSIGNAL_GREEN",
    6: "TRAFFICSIGNAL_LEFT",
    7: "TRAFFICSIGNAL_GREEN_LEFT",
}
