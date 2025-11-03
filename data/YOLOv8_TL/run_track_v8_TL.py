from ultralytics import YOLO
import numpy as np
import cv2
import os
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_tl import Comm
from utils.check_tl import Check
# from utils.check import draw_test_frame

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT, MAIN_SERVER_TCP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2, TEST_VID_DIR
from conf.config import LIGHT_CLS_DIR

import uuid

# =========================
# Option
# =========================
DEBUG_SAVE          = True    # Debug Save (images will save)
SAVE_PREVIEW        = False   # Check box to all classes which exists in one frame
SAVE_SELECTED       = True    # Only detected objects are checked
SAVE_CROP           = False   # Crop image save
DRAW_ROI_ON_SAVE    = False   # Draw ROI line to your saved image
SHOW_LABEL_ON_SAVE  = False   # Mark the label to your save image

DEBUG_DIR           = "/data/debug_tl"
PREVIEW_SUBDIR      = "preview"
SELECTED_SUBDIR     = "selected"
CROP_SUBDIR         = "crop"


def ensure_dirs():
    if not DEBUG_SAVE:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    if SAVE_PREVIEW:
        os.makedirs(os.path.join(DEBUG_DIR, PREVIEW_SUBDIR), exist_ok=True)
    if SAVE_SELECTED:
        os.makedirs(os.path.join(DEBUG_DIR, SELECTED_SUBDIR), exist_ok=True)
    if SAVE_CROP:
        os.makedirs(os.path.join(DEBUG_DIR, CROP_SUBDIR), exist_ok=True)


def gen_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def xywh_to_xyxy(xywh, W, H):
    cx, cy, w, h = xywh
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2


def main():
    # model = YOLO(AUG_V8_WEIGHTS_DIR_V1_2 + '/weights/best.pt')
    model = YOLO("/data/YOLOv8_TL/best.pt")
    model.fuse()
    
    # Upper -> realtime, Lower -> video test
    dataset = LCD_LoadStreams(sources="192.168.10.116", vid_stride=1, buffer=True)
    # dataset = LoadImagesAndVideos("/data/Integration_test.mp4", vid_stride=1)

    obj_track_dict = {} 
    sent_id_cls = []      
    sent_cls = []         
    ch = Check()
    fnum = 0

    global send_once
    ensure_dirs()

    for source, imgs, bs, count in dataset:
        np_img = np.array(imgs[0], dtype=np.uint8)
        if np_img is None:
            fnum += 1
            continue

        H, W = np_img.shape[:2]

        results = model.track(
            np_img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            classes=[2, 3, 4, 5],
            verbose=False
        )

        if len(results) == 0:
            fnum += 1
            continue

        results[0].names = LIGHT_CLS_DIR

        preview_frame = results[0].plot(labels=False, conf=False)

        if DEBUG_SAVE and SAVE_PREVIEW:
            cv2.imwrite(
                os.path.join(DEBUG_DIR, PREVIEW_SUBDIR, f"{now_str()}_preview_f{fnum}.jpg"),
                preview_frame
            )

        base_frame = np_img.copy()

        for r in results:
            if r.boxes.id is None:
                continue

            r.names = LIGHT_CLS_DIR

            id_list = list(map(int, r.boxes.id.tolist()))
            cls_list = list(map(int, r.boxes.cls.tolist()))
            full_list = [f'{id}{cls:03d}' for id, cls in zip(id_list, cls_list)]

            for idx, (id_, cls, conf, xywh) in enumerate(zip(
                id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh
            )):
                cls = int(cls)
                conf = float(conf)
                xywh = xywh.tolist()

                id_cls = int(f'{id_}{cls:03d}')

                obj_track_dict[id_cls] = {
                    'cls_name': r.names[cls],
                    'cls': cls,
                    'conf': conf,
                    'xywh': xywh
                }

                if not ch.check_valid(cls, xywh, conf):
                    continue
                if not ch.check_count(id_, str(id_cls)):
                    continue

                if obj_track_dict[id_cls]['cls'] == 2:
                    continue

                if send_once and (id_cls in sent_id_cls):
                    continue

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

                sel_frame = base_frame.copy()

                if DRAW_ROI_ON_SAVE:
                    sel_frame = ch.draw_zone(sel_frame)

                x1, y1, x2, y2 = xywh_to_xyxy(obj_track_dict[id_cls]['xywh'], W, H)
                cv2.rectangle(sel_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if SHOW_LABEL_ON_SAVE:
                    label = f"{obj_track_dict[id_cls]['cls_name']} {obj_track_dict[id_cls]['conf']:.2f} id{id_}"
                    cv2.putText(sel_frame, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                com.upload_to_ftp(sel_frame, obj_track_dict[id_cls]['cls_name'], img_file_name)

                if DEBUG_SAVE and SAVE_SELECTED:
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, SELECTED_SUBDIR,
                                     f"{now_str()}_sel_f{fnum}_idcls{id_cls}_{obj_track_dict[id_cls]['cls_name']}_{obj_track_dict[id_cls]['conf']:.2f}.jpg"),
                        sel_frame
                    )
                if DEBUG_SAVE and SAVE_CROP:
                    crop = base_frame[y1:y2, x1:x2].copy()
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, CROP_SUBDIR,
                                     f"{now_str()}_crop_f{fnum}_idcls{id_cls}_{obj_track_dict[id_cls]['cls_name']}_{obj_track_dict[id_cls]['conf']:.2f}.jpg"),
                        crop
                    )

                if send_once:
                    sent_id_cls.append(id_cls)
                    sent_cls.append(cls)

            for key in list(obj_track_dict.keys()):
                key_str = str(key)
                if key_str not in full_list:
                    del obj_track_dict[key]
                    if key in sent_id_cls:
                        sent_id_cls.remove(key)

        fnum += 1

        # cv2.imshow("TL Tracking Preview", preview_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break


if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()
