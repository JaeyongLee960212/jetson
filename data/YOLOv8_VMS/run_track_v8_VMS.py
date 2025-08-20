from ultralytics import YOLO
import numpy as np
import cv2
import hashlib
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_vms import Comm
import os

from utils.check_vms import Check
#from utils.check import draw_test_frame
from utils.OCR import OCR

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT, MAIN_SERVER_TCP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2, TEST_VID_DIR

import uuid

#def gen_hash(s):
#    hash_obj = hashlib.sha256()
#    hash_obj.update(s.encode('utf-8'))
#    return hash_obj.hexdigest()[:20] # duplicate is highly unlikely .... might be? 

def gen_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main():
    # model = YOLO(AUG_V8_WEIGHTS_DIR_V1_2+'/models/epoch100.pt')  # Load an official Detect model
    model = YOLO("/data/YOLOv8_VMS/VMS_best_250701.pt")
    model.fuse() # fusion conv2d and batchNorm2d 

    #dataset = LCD_LoadStreams(sources = "192.168.10.116", vid_stride = 1, buffer = True)
    dataset = LoadImagesAndVideos("/data/YOLOv8_VMS/Send_Test_VMS_LCS.mp4", vid_stride = 1)

    # obj_track_dict ex) {0: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[10,20,30,40]}, 2: {'cls':0, 'cls_name':r.names[int(cls)],'conf':0.9, 'xywh':[133,203,303,430]} }
    obj_track_dict = {}
    sent_id = []
    ch = Check()
    #ocr = OCR()
    fnum = 0
    output_dir = '/data/test/vms_ocr'
    os.makedirs(output_dir, exist_ok=True)
    
    global send_once

    for source, imgs, bs, count in dataset: # capture one frame \
        np_img = np.array(imgs[0], dtype=np.uint8)
        results = model.track(np_img, persist=True, tracker = "bytetrack.yaml", conf=0.3, verbose=False)         # track  # https://docs.ultralytics.com/ko/reference/engine/model/#ultralytics.engine.model.Model_
        annotated_frame = results[0].plot()                                             # Visualize the result on the frame  -->  /home/keti/yolov8/ultralytics/ultralytics/engine/results.py               
        
        # annotated_frame = draw_test_frame(annotated_frame, None)
        for r in results:    
            if r.boxes.id is not None:
                id_list = list(map(int, r.boxes.id.tolist())) 
                
                # mapping { id : {cls : int, cls_name : str, conf : float , xywh : list} and appends dictionary
                for idx, (id, cls, conf, xywh) in enumerate(zip(id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh)): # xywh : --> xy : bbox ceneter x,y wh : box wh ,,,,, if use r.boxes.xyxy : bbox x1,y1,x2,y2
                    obj_track_dict[id] = {
                        'cls_name': r.names[int(cls)],
                        'cls': int(cls),
                        'conf': float(conf),
                        'xywh': xywh.tolist()
                    }
                    
                    x_center, y_center, width, height = map(int, xywh)
                    print(f"[FRAME {fnum}] ID:{id} | Class:{r.names[int(cls)]} | "f"Center:({x_center},{y_center}) | Width:{width}px | Height:{height}px | Conf:{conf}")        
                                
                    if not ch.check_valid(cls, xywh, conf):
                        continue

                    if not ch.check_count(id):
                        continue
                    
                    #send once  
                    if (send_once == True and id not in sent_id):

                        # annotated_frame crop
                        for obj_id, obj_info in obj_track_dict.items():
                            xywh = obj_info['xywh']
                            x_center, y_center, width, height = map(int, xywh)
                
                            w = int(width) # adjust cropped image margin
                            h = int(height*0.9)
                            
                            xmin = max(0, x_center - w // 2)
                            ymin = max(0, y_center - h // 2)
                            xmax = min(np_img.shape[1], x_center + w // 2)
                            ymax = min(np_img.shape[0], y_center + h // 2)

                            cropped_img = np_img[ymin:ymax, xmin:xmax]
                        
                        # gen detect time & img file name
                        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                        img_file_name = gen_uuid(detect_time) + '.jpg'
                        
                        # save OCR image
                        #cv2.imwrite(f'/data/test/vms_ocr{fnum}.jpg', cropped_img)
                        #ocr_text = ocr.OCR(f'/data/test/vms_ocr_{fnum}.jpg')
                        
                        cv2.imwrite(f'/data/test/vms_ocr/{img_file_name}', cropped_img)
                        #ocr_text = ocr.OCR(f'/data/test/vms_ocr/{img_file_name}')

                        # convert msg json type
                        msg = com.pack_data(obj_track_dict[id]['cls'], \
                                            obj_track_dict[id]['cls_name'], \
                                            obj_track_dict[id]['conf'], \
                                            obj_track_dict[id]['xywh'], \
                                            fnum, \
                                            img_file_name, \
                                            detect_time
                                            )
                        
                        # send msg to nuvo & upload img to nuvo ftp server 
                        com.send_data_to_udp(msg)
                        com.upload_to_ftp(annotated_frame, obj_track_dict[id]['cls_name'], img_file_name)

                        #open(f'/data/test/{img_file_name[:-4]}.txt', 'w').write(msg)
                        open(f'/data/test/{img_file_name[:-4]}.txt', 'w').write(msg)
                        #cv2.imwrite(f'/data/test/vms_{img_file_name}.jpg', annotated_frame)
                        cv2.imwrite(f'/data/test/{r.names[int(cls)]}_{float(conf)}.jpg', annotated_frame)

                        if (send_once == True):
                            sent_id.append(id)
                            #send_once = False
                    
                # delete key. if id is not in frame.  
                for key in list(obj_track_dict.keys()):
                    if key not in id_list:
                        del obj_track_dict[key]

                        if key in sent_id:
                            sent_id.remove(key)

                # print("obj_track_dict: ", obj_track_dict[id])          
                               
        fnum += 1       
        
        #cv2.imshow("Tracking", annotated_frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #   break
    
if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()

#def main():
#    model = YOLO('/jetson-inference/data/road_infra_yolov8/test_best2.pt')  # Load an official Detect model
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
