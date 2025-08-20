from ultralytics import YOLO
import numpy as np
import cv2
import hashlib
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_tl import Comm
from utils.check_tl import Check
from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2
from conf.config import LIGHT_CLS_DIR
import json
import os

import uuid

#def gen_hash(s):
#    hash_obj = hashlib.sha256()
#    hash_obj.update(s.encode('utf-8'))
#    return hash_obj.hexdigest()[:20] # duplicate is highly unlikely .... might be? 

def gen_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main():
    # 모델 로드
    model = YOLO("/data/YOLOv8_TL/best.pt")
    model.fuse()
    dataset = LoadImagesAndVideos("/data/YOLOv8_TL/Blink_test_1.mp4", vid_stride=1)

    obj_track_dict = {}
    sent_id_cls = []
    ch = Check()
    fnum = 0

    global send_once

    # 로그 파일 오픈
    if not os.path.exists("/data/test1/all_frames_1"):
        os.makedirs("/data/test1/all_frames_1")
    log_txt_path = "/data/test1/full_log_1.txt"
    log_json_path = "/data/test1/full_log_1.json"
    log_txt = open(log_txt_path, "w")
    log_json_list = []

    for source, imgs, bs, count in dataset:
        np_img = np.array(imgs[0], dtype=np.uint8)
        results = model.track(np_img, persist=True, tracker="bytetrack.yaml", conf=0.3, classes=[2,3,4,5])

        results[0].names = LIGHT_CLS_DIR
        annotated_frame = results[0].plot()

        for r in results:
            if r.boxes.id is not None:
                r.names = LIGHT_CLS_DIR
                id_list = list(map(int, r.boxes.id.tolist()))
                cls_list = list(map(int, r.boxes.cls.tolist()))
                full_list = [f'{id}{cls:03d}' for id, cls in zip(id_list, cls_list)]

                for idx, (id, cls, conf, xywh) in enumerate(zip(id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh)):
                    id_cls = int(f'{id}{int(cls):03d}')
                    obj_track_dict[id_cls] = {
                        'cls_name': r.names[int(cls)],
                        'cls': int(cls),
                        'conf': float(conf),
                        'xywh': xywh.tolist()
                    }

                    x_center, y_center, width, height = map(int, xywh)
                    class_name = r.names[int(cls)]

                    # 로그 텍스트 작성
                    log_line = (
                        f"[FRAME {fnum}] ID:{id} | Class:{class_name} | "
                        f"Center:({x_center},{y_center}) | Width:{width}px | Height:{height}px | Conf:{float(conf):.2f}\n"
                    )
                    log_txt.write(log_line)

                    # JSON-style 로그 저장
                    log_json_list.append({
                        "frame": fnum,
                        "id": int(id),
                        "class": class_name,
                        "cls_id": int(cls),
                        "conf": float(conf),
                        "xywh": xywh.tolist()
                    })

                    if not ch.check_valid(cls, xywh, conf):
                        continue
                    if not ch.check_count(id, str(id_cls)):
                        continue

                    if send_once == True and id_cls not in sent_id_cls:
                        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                        img_file_name = gen_uuid(detect_time) + '.jpg'

                        msg = com.pack_data(
                            obj_track_dict[id_cls]['cls'],
                            obj_track_dict[id_cls]['cls_name'],
                            obj_track_dict[id_cls]['conf'],
                            obj_track_dict[id_cls]['xywh'],
                            fnum,
                            detect_time,
                            img_file_name
                        )

                        com.send_data_to_udp(msg)
                        com.upload_to_ftp(annotated_frame, obj_track_dict[id_cls]['cls_name'], img_file_name)

                        cv2.imwrite(f'/data/test1/{class_name}_{float(conf):.2f}.jpg', ch.draw_zone(annotated_frame))

                        sent_id_cls.append(id_cls)

                # 트래킹 id 정리
                for key in list(obj_track_dict.keys()):
                    if str(key) not in full_list:
                        del obj_track_dict[key]
                        if key in sent_id_cls:
                            sent_id_cls.remove(key)

        # 모든 프레임 이미지 저장
        cv2.imwrite(f"/data/test1/all_frames_1/frame_{fnum:05d}.jpg", annotated_frame)

        fnum += 1

    log_txt.close()
    with open(log_json_path, "w") as fjson:
        json.dump(log_json_list, fjson, indent=4)

if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()
