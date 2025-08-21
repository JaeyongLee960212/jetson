from ultralytics import YOLO

#if pytorch version is higher than 2.6 import this
#import torch
#from torch.nn.modules.container import Sequential, ModuleList
#from torch.nn.modules.conv import Conv2d
#from torch.nn.modules.batchnorm import BatchNorm2d
#from torch.nn.modules.activation import SiLU
#from torch.nn.modules.upsampling import Upsample

#from ultralytics.nn.tasks import DetectionModel
#from ultralytics.nn.modules.conv import Conv
#from ultralytics.nn.modules.block import C2f, C3, SPPF
#from ultralytics.nn.modules.head import Detect
#from ultralytics.nn.modules.block import Bottleneck
#pytorch > 2.6 이면 /opt/venv/lib/python3.12/site-packages/ultralytics/nn/tasks.py 가서 torch.load 수정해야댐 shiver

import numpy as np
import cv2
import hashlib
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_lcs import Comm

from utils.check_lcs import Check
#from utils.check import draw_test_frame
import os

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT, MAIN_SERVER_TCP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2, TEST_VID_DIR
from conf.config import LCS_CLS_DIR

from collections import deque, Counter

#torch.serialization.add_safe_globals([
    # torch native
#    Sequential,
#    ModuleList,
#    Conv2d,
#    BatchNorm2d,
#    SiLU,
#    Upsample,
    
    # ultralytics
#    DetectionModel,
#    Conv,
#    C2f,
#    C3,
#    SPPF,
#    Detect,
#    Bottleneck,
#])
import uuid

#def gen_hash(s):
#    hash_obj = hashlib.sha256()
#    hash_obj.update(s.encode('utf-8'))
#    return hash_obj.hexdigest()[:20] # duplicate is highly unlikely .... might be? 

def gen_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main():
    model = YOLO("/data/YOLOv8_LCS/LCS_best.pt")
    model.fuse()

    dataset = LoadImagesAndVideos("/data/YOLOv8_LCS/Send_Test_VMS_LCS.mp4", vid_stride=1)

    ch = Check()
    first_keys = []
    second_keys = []
    sent_id = []
    obs = {}

    send_trigger = False
    save_obs = deque([], maxlen=5)  # None 대신 빈 리스트
    save_frames = deque([], maxlen=5)
    fnum = 0
    no_obs = 0
    patience = 8

    global send_once

    for source, imgs, bs, count in dataset:
        np_img = np.array(imgs[0], dtype=np.uint8)
        results = model.track(np_img, conf=0.3, classes=[0, 1, 2], verbose=False)

        results[0].names = LCS_CLS_DIR
        annotated_frame = results[0].plot()
        fnum += 1

        obs_list = {}
        pframes = {}
        id_list = []

        for r in results:
            results[0].names = LCS_CLS_DIR
            cls_list = list(map(int, r.boxes.cls.tolist()))
            conf_list = list(map(float, r.boxes.conf.tolist()))

            for idx, (cls, conf, xywh) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh)):
                id = (f'{fnum}{idx}{int(cls)}')

                obs[id] = {
                    'cls_name': r.names[int(cls)],
                    'conf': float(conf),
                    'xywh': xywh.tolist()
                }

                x_center, y_center, width, height = map(int, xywh)
                print(f"[FRAME {fnum}] ID:{id} | Class:{r.names[int(cls)]} | "
                      f"Center:({x_center},{y_center}) | Width:{width}px | Height:{height}px | conf:{conf}")

                if (ch.check_valid(cls, xywh, conf)) and (len(obs.keys()) >= 2):
                    obs_list = dict(sorted(obs.items(), key=lambda item: item[1]['conf'], reverse=True))
                    obs_list = dict(list(obs_list.items())[:2])
                    obs_list = dict(sorted(obs_list.items(), key=lambda item: item[1]['xywh'][0]))

                    list(obs_list.items())[0][1]['cls_name'] += '_1'
                    list(obs_list.items())[1][1]['cls_name'] += '_2'

                    pframes[fnum] = obs_list

        if len(pframes) != 0:
            save_obs.append(pframes)
            save_frames.append(annotated_frame)

        if (len(obs) == 0) and (len(save_obs) > 0):
            no_obs += 1
            if no_obs > patience:
                send_trigger = True
        else:
            no_obs = 0

        for key in list(obs.keys()):
            if key not in cls_list:
                del obs[key]

        if send_trigger:
            send_trigger = False

            for frame_dict in save_obs:
                if frame_dict is None:
                    continue
                for ids in frame_dict.values():
                    id_list.append(list(ids.keys()))

            if not id_list:
                continue

            first = [k[0] for k in id_list]
            second = [k[1] for k in id_list]

            first_keys = Counter(first)
            second_keys = Counter(second)

            most_common_first = first_keys.most_common(1)[0][0]
            most_common_second = second_keys.most_common(1)[0][0]

            last_index = None
            for index, kkk in enumerate(id_list):
                if most_common_first in kkk and most_common_second in kkk:
                    last_index = index

            if last_index is not None:
                frame_data = save_obs[last_index]
                frame = save_frames[last_index]
                for key, obj in frame_data.items():
                    for id, data in obj.items():
                        if id in sent_id:
                            continue  

                        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                        img_file_name = gen_uuid(detect_time) + '.jpg'

                        msg = com.pack_data(int(id[-1]),
                                            data.get('cls_name', 'unknown'),
                                            data.get('conf', 0.0),
                                            data.get('xywh', [0, 0, 0, 0]),
                                            fnum,
                                            detect_time,
                                            img_file_name)

                        com.send_data_to_udp(msg)
                        com.upload_to_ftp(frame, data.get('cls_name', 'unknown'), img_file_name)

                        result_name = f"{str(fnum)}_{data.get('cls_name', 'unknown')}_{float(data.get('conf', 0.0))}"
                        os.makedirs('results', exist_ok=True)
                        open(f'data/test/{result_name}.txt', 'w').write(msg)
                        cv2.imwrite(f'data/test/{result_name}.jpg', frame)
                        print(msg)

                        # 전송한 객체는 기록
                        sent_id.append(id)

if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()



#def main():
#    model = YOLO('/jetson-inference/data/road_infra_yolov8/test_best3.pt')  # Load an official Detect model
#    # model = YOLO('./models_v8/model_scratch.pt')  # Load an official Detect model
#    model.fuse() # fusion conv2d and batchNorm2d 
#
#    dataset = LCD_LoadStreams(sources = "192.168.10.116", vid_stride = 1, buffer = True)
#    #dataset = LoadImagesAndVideos("/jetson-inference/data/road_infra_yolov8/ts-test1.mp4", vid_stride = 1)
#
#    # obj_track_dict ex) {0: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[10,20,30,40]}, 2: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[133,203,303,430]} }
#    obj_track_dict = {}
#    sent_id = []
#    ch = Check()
#
#    while True: # run
#        for source, imgs, bs, count in dataset: # capture one frame \
#            
#            np_img = np.array(imgs[0], dtype=np.uint8)
#            results = model.track(np_img, persist=True, tracker = "bytetrack.yaml")         # track  # https://docs.ultralytics.com/ko/reference/engine/model/#ultralytics.engine.model.Model_
#            annotated_frame = results[0].plot()                                             # Visualize the result on the frame  -->  /home/keti/yolov8/ultralytics/ultralytics/engine/results.py               
#            
#            for r in results:    
#                if r.boxes.id is not None:
#                    id_list = list(map(int, r.boxes.id.tolist())) 
#                    
#                    # mapping { id : {cls : int, cls_name : str, conf : float , xywh : list} and appends dictionary
#                    for idx, (id, cls, conf, xywh) in enumerate(zip(id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh)): # xywh : --> xy : bbox ceneter x,y wh : box wh ,,,,, if use r.boxes.xyxy : bbox x1,y1,x2,y2
#                        obj_track_dict[id] = {
#                            'cls_name': r.names[int(cls)],
#                            'cls': int(cls),
#                            'conf': float(conf),
#                            'xywh': xywh.tolist()
#                        }
#                        if not ch.check_valid(cls, xywh, conf):
#                            continue
#
#                        if not ch.check_count(id):
#                            continue
#
#                        #send once  
#                        if (send_once == True and id not in sent_id): #and check_valid()
#
#                            # gen detect time & img file name
#                            detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
#                            img_file_name = gen_hash(detect_time) + '.jpg'
#
#                            # convert msg json type
#                            msg = com.pack_data(obj_track_dict[id]['cls'], \
#                                                obj_track_dict[id]['cls_name'], \
#                                                obj_track_dict[id]['conf'], \
#                                                obj_track_dict[id]['xywh'], \
#                                                count, \
#                                                detect_time, \
#                                                img_file_name)
#                            
#                            # send msg to nuvo & upload img to nuvo ftp server 
#                            com.send_data_to_udp(msg)
#                            com.upload_to_ftp(annotated_frame, obj_track_dict[id]['cls_name'], img_file_name)
#                            sent_id.append(id)
#                        
#                        # draw_test_frame(annotated_frame, xywh.tolist())
#
#                    # delete key. if id is not in frame.  
#                    for key in list(obj_track_dict.keys()):
#                        if key not in id_list:
#                            del obj_track_dict[key]
#
#                            if key in sent_id:
#                                sent_id.remove(key)
#    
#                    # print("obj_track_dict: ", obj_track_dict[id])
#                                       
#            #cv2.imshow("Tracking", annotated_frame)
#            if cv2.waitKey(1) & 0xFF == ord("q"):
#                break
#            
#if __name__ == "__main__":
#    send_once = True
#    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
#    main()














































# if __name__ == "__main__":
    
#     model = YOLO('yolov8n.pt')  # Load an official Detect model
#     model.fuse()

#     byte_tracker = sv.ByteTrack()
#     annotator = sv.BoxAnnotator()
#     cap = Stream_Vision_Cam()  # store video capture object
#     cap.setup(cam_ip='192.168.10.41', width=1280, height=720)
#     while True:
#         success, frame = cap.capture()
#         results = model.track(frame, persist=True)
#         # #Visualize the result on the frame
#         annotated_frame = results.plot()

#         cv2.imshow("Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
