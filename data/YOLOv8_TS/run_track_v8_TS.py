from ultralytics import YOLO
import numpy as np
import cv2
import os
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils import Comm
from utils.check_ts import Check
# from utils.check import draw_test_frame 

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT
from conf.config import SIGN_CLS_DIR, U1_CLASS_MAPPING_DIR

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

DEBUG_DIR         = "/data/debug"  
PREVIEW_SUBDIR    = "preview"
SELECTED_SUBDIR   = "selected"
CROP_SUBDIR       = "crop"


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


def gen_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def xywh_to_xyxy(xywh, W, H):
    """(cx, cy, w, h) -> (x1, y1, x2, y2) with clamp"""
    cx, cy, w, h = xywh
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2


def main():
    model = YOLO("/data/YOLOv8_TS/ts_model_best2.pt")
    model.fuse()  # Conv+BN fusion

    dataset = LCD_LoadStreams(sources="192.168.10.116", vid_stride=1, buffer=True)
    # dataset = LoadImagesAndVideos("/data/Integration_test.mp4", vid_stride=1)

    obj_track_dict = {}
    sent_id = []
    sent_cls = []
    ch = Check()
    fnum = 0

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
            classes=list(range(0, 82)),
            verbose=False
        )
        if len(results) == 0:
            fnum += 1
            continue

        results[0].names = SIGN_CLS_DIR
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

            r.names = SIGN_CLS_DIR
            id_list = list(map(int, r.boxes.id.tolist()))

            for (tid, cls, conf, xywh) in zip(
                id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh
            ):
                cls = int(cls)
                conf = float(conf)
                xywh = xywh.tolist()

                cls_name = r.names[cls]
                subtype = U1_CLASS_MAPPING_DIR.get(cls_name, None)

                obj_track_dict[tid] = {
                    'cls_name': cls_name,
                    'cls': cls,
                    'conf': conf,
                    'xywh': xywh
                }

                if not ch.check_valid(cls, xywh, conf):
                    continue
                if not ch.check_count(tid):
                    continue
                if send_once and (tid in sent_id):
                    continue

                detect_time_h = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                detect_time_f = now_str()
                img_file_name = gen_uuid(detect_time_h) + ".jpg"

                msg = com.pack_data(
                    subtype,
                    cls_name,
                    conf,
                    xywh,
                    fnum,
                    detect_time_h,
                    img_file_name
                )
                com.send_data_to_udp(msg)

                sel_frame = base_frame.copy()

                if DRAW_ROI_ON_SAVE:
                    sel_frame = ch.draw_zone(sel_frame)

                x1, y1, x2, y2 = xywh_to_xyxy(xywh, W, H)

                cv2.rectangle(sel_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if SHOW_LABEL_ON_SAVE:
                    label = f"{cls_name} {conf:.2f} id{tid}"
                    cv2.putText(sel_frame, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                com.upload_to_ftp(sel_frame, cls_name, img_file_name)

                if DEBUG_SAVE and SAVE_SELECTED:
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, SELECTED_SUBDIR,
                                     f"{detect_time_f}_sel_f{fnum}_id{tid}_{cls_name}_{conf:.2f}.jpg"),
                        sel_frame
                    )

                if DEBUG_SAVE and SAVE_CROP:
                    crop = base_frame[y1:y2, x1:x2].copy()
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, CROP_SUBDIR,
                                     f"{detect_time_f}_crop_f{fnum}_id{tid}_{cls_name}_{conf:.2f}.jpg"),
                        crop
                    )

                if send_once:
                    sent_id.append(tid)
                    sent_cls.append(cls)

            for key in list(obj_track_dict.keys()):
                if key not in id_list:
                    del obj_track_dict[key]
                    if key in sent_id:
                        sent_id.remove(key)

        fnum += 1

if __name__ == "__main__":
    send_once = True
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()
