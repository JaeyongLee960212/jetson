from ultralytics import YOLO
import numpy as np
import cv2
import hashlib
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils import Comm

from utils.check_ts import Check
#from utils.check import draw_test_frame

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT, MAIN_SERVER_TCP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2, TEST_VID_DIR
from conf.config import SIGN_CLS_DIR, U1_CLASS_MAPPING_DIR

import uuid

#def gen_hash(s):
#    hash_obj = hashlib.sha256()
#    hash_obj.update(s.encode('utf-8'))
#    return hash_obj.hexdigest()[:20] # duplicate is highly unlikely .... might be? 

def gen_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main():
    # model = YOLO(AUG_V8_WEIGHTS_DIR_V1_2+'/weights/best.pt')  # Load an official Detect model
    model = YOLO("/data/YOLOv8_TS/ts_model_best2.pt")
    model.fuse() # fusion conv2d and batchNorm2d 

    #dataset = LCD_LoadStreams(sources = "192.168.10.116", vid_stride = 1, buffer = True)
    dataset = LoadImagesAndVideos("/data/YOLOv8_TS/ts-test1.mp4", vid_stride = 1)

    # obj_track_dict ex) {0: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[10,20,30,40]}, 2: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[133,203,303,430]} }
    obj_track_dict = {}
    sent_id = []
    ch = Check()
    fnum = 0
    sent_cls = []

    for source, imgs, bs, count in dataset: # capture one frame \
        np_img = np.array(imgs[0], dtype=np.uint8)
        results = model.track(np_img, persist=True, tracker = "bytetrack.yaml", conf=0.3,
                              classes = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                         30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                         40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                         50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                         60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                         70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                         80, 81],
                              verbose=False)         # track  # https://docs.ultralytics.com/ko/reference/engine/model/#ultralytics.engine.model.Model_
        
        results[0].names = SIGN_CLS_DIR
        annotated_frame = results[0].plot()                                             # Visualize the result on the frame  -->  /home/keti/yolov8/ultralytics/ultralytics/engine/results.py               
        #print(U1_ClASS_MAPPING_DIR.get(results[0].names, None)
        #subtype = U1_ClASS_MAPPING_DIR.get(results[0].names, None)
        for cls_id in results[0].boxes.cls:
            class_name = results[0].names[int(cls_id)]  # 예: 'SAFETY_SIGN_STOP'
            subtype = U1_CLASS_MAPPING_DIR.get(class_name, None)

            print(f"Class ID: {cls_id}, Class Name: {class_name}, Subtype: {subtype}")
        
        # annotated_frame = draw_test_frame(annotated_frame, None)
        for r in results:    
            if r.boxes.id is not None:
                
                r.names = SIGN_CLS_DIR
                id_list = list(map(int, r.boxes.id.tolist())) 
                
                # mapping { id : {cls : int, cls_name : str, conf : float , xywh : list} and appends dictionary
                for idx, (id, cls, conf, xywh) in enumerate(zip(id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh)): # xywh : --> xy : bbox ceneter x,y wh : box wh ,,,,, if use r.boxes.xyxy : bbox x1,y1,x2,y2
                    obj_track_dict[id] = {
                        'cls_name': r.names[int(cls)],
                        'cls': int(cls),
                        'conf': float(conf),
                        'xywh': xywh.tolist()
                    }
                    
                    if not ch.check_valid(cls, xywh, conf):
                        continue

                    if not ch.check_count(id):
                        continue
                    
                    #send once  
                    if (send_once == True and int(id) not in sent_id): #and check_valid()                  
                        
                        # gen detect time & img file name
                        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                        img_file_name = gen_uuid(detect_time) + '.jpg'
                        
                        # convert msg json type
                        msg = com.pack_data(subtype, \
                                            obj_track_dict[id]['cls_name'], \
                                            obj_track_dict[id]['conf'], \
                                            obj_track_dict[id]['xywh'], \
                                            fnum, \
                                            detect_time, \
                                            img_file_name)
                        
                        # send msg to nuvo & upload img to nuvo ftp server 
                        com.send_data_to_udp(msg)
                        com.upload_to_ftp(annotated_frame, obj_track_dict[id]['cls_name'], img_file_name)
                        
                        #result_name = f'detected_{detect_time}_{id}'
                        #import os
                        #os.makedirs('results', exist_ok=True)
                        #open(f'results/{result_name}.txt', 'w').write(msg)
                        #cv2.imwrite(f'/data/results/{result_name}.jpg', annotated_frame)
                        cv2.imwrite(f'/data/test/{r.names[int(cls)]}_{float(conf)}.jpg', ch.draw_zone(annotated_frame))
                        
                        if (send_once == True):
                            sent_id.append(id)
                            sent_cls.append(int(cls))
                    
                # delete key. if id is not in frame.  
                for key in list(obj_track_dict.keys()):
                    if key not in id_list:
                        del obj_track_dict[key]

                        if key in sent_id:
                            sent_id.remove(key)
                
        fnum += 1       
        
        # cv2.imshow("Tracking", ch.draw_zone(annotated_frame)) # check roi box
        #cv2.imshow("Tracking", annotated_frame)

        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #   break
    
if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()

#def main():
#    model = YOLO('/jetson-inference/data/road_infra_yolov8/best.pt')  # Load an official Detect model
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
