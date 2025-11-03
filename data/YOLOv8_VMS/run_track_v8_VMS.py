from ultralytics import YOLO
import numpy as np
import cv2
import os
from datetime import datetime
from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_vms import Comm

from utils.check_vms import Check
# from utils.check import draw_test_frame
# from utils.OCR import OCR  # 필요 시 사용

from conf.config import MAIN_SERVER_IP
from conf.config import MAIN_SERVER_UDP_PORT, MAIN_SERVER_TCP_PORT
from conf.config import AUG_V8_WEIGHTS_DIR_V1_2, TEST_VID_DIR

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

DEBUG_DIR           = "/data/debug_vms"
PREVIEW_SUBDIR      = "preview"
SELECTED_SUBDIR     = "selected"
CROP_SUBDIR         = "crop"

OCR_CROP_DIR        = "/data/test/vms_ocr"  # 기존 경로 유지


def ensure_dirs():
    if DEBUG_SAVE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        if SAVE_PREVIEW:
            os.makedirs(os.path.join(DEBUG_DIR, PREVIEW_SUBDIR), exist_ok=True)
        if SAVE_SELECTED:
            os.makedirs(os.path.join(DEBUG_DIR, SELECTED_SUBDIR), exist_ok=True)
        if SAVE_CROP:
            os.makedirs(os.path.join(DEBUG_DIR, CROP_SUBDIR), exist_ok=True)
    os.makedirs(OCR_CROP_DIR, exist_ok=True)
    os.makedirs("/data/test", exist_ok=True)


def gen_uuid(s):
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


def safe_crop(img, xywh):
    H, W = img.shape[:2]
    cx, cy, w, h = map(int, xywh)
    w_adj = int(w)
    h_adj = int(h * 0.9)
    xmin = max(0, cx - w_adj // 2)
    ymin = max(0, cy - h_adj // 2)
    xmax = min(W, cx + w_adj // 2)
    ymax = min(H, cy + h_adj // 2)
    return img[ymin:ymax, xmin:xmax].copy(), (xmin, ymin, xmax, ymax)


def main():
    # model = YOLO(AUG_V8_WEIGHTS_DIR_V1_2 + '/models/epoch100.pt')
    model = YOLO("/data/YOLOv8_VMS/VMS_best_250701.pt")
    model.fuse()  # Conv+BN fusion

    dataset = LCD_LoadStreams(sources="192.168.10.116", vid_stride=1, buffer=True)
    # dataset = LoadImagesAndVideos("/data/YOLOv8_VMS/Send_Test_VMS_LCS.mp4", vid_stride=1)

    obj_track_dict = {}
    sent_id = []
    ch = Check()
    # ocr = OCR()
    fnum = 0

    global send_once
    ensure_dirs()

    for source, imgs, bs, count in dataset:  # capture one frame
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
            verbose=False
        )
        if len(results) == 0:
            fnum += 1
            continue

        preview_frame = results[0].plot(labels=False, conf=False)

        if DEBUG_SAVE and SAVE_PREVIEW:
            cv2.imwrite(
                os.path.join(DEBUG_DIR, PREVIEW_SUBDIR, f"{now_str()}_preview_f{fnum}.jpg"),
                preview_frame
            )

        base_frame = np_img.copy()

        # annotated_frame = draw_test_frame(preview_frame, None)
        for r in results:
            if r.boxes.id is None:
                continue

            id_list = list(map(int, r.boxes.id.tolist()))

            for idx, (tid, cls, conf, xywh) in enumerate(
                zip(id_list, r.boxes.cls, r.boxes.conf, r.boxes.xywh)
            ):
                cls = int(cls)
                conf = float(conf)
                xywh = xywh.tolist()

                obj_track_dict[tid] = {
                    'cls_name': r.names[int(cls)],
                    'cls': int(cls),
                    'conf': conf,
                    'xywh': xywh
                }

                if not ch.check_valid(cls, xywh, conf):
                    continue
                if not ch.check_count(tid):
                    continue
                if obj_track_dict[tid]['cls'] == 1:
                    continue

                if send_once and tid in sent_id:
                    continue

                crop_img, (xmin, ymin, xmax, ymax) = safe_crop(base_frame, xywh)
                detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                img_file_name = gen_uuid(detect_time) + '.jpg'
                cv2.imwrite(os.path.join(OCR_CROP_DIR, img_file_name), crop_img)

                msg = com.pack_data(
                    obj_track_dict[tid]['cls'],
                    obj_track_dict[tid]['cls_name'],
                    obj_track_dict[tid]['conf'],
                    obj_track_dict[tid]['xywh'],
                    fnum,
                    img_file_name,
                    detect_time
                )

                com.send_data_to_udp(msg)

                sel_frame = base_frame.copy()

                if DRAW_ROI_ON_SAVE:
                    sel_frame = ch.draw_zone(sel_frame)

                x1, y1, x2, y2 = xywh_to_xyxy(obj_track_dict[tid]['xywh'], W, H)
                cv2.rectangle(sel_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if SHOW_LABEL_ON_SAVE:
                    label = f"{obj_track_dict[tid]['cls_name']} {obj_track_dict[tid]['conf']:.2f} id{tid}"
                    cv2.putText(sel_frame, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                com.upload_to_ftp(sel_frame, obj_track_dict[tid]['cls_name'], img_file_name)

                open(f'/data/test/{img_file_name[:-4]}.txt', 'w').write(msg)

                if DEBUG_SAVE and SAVE_SELECTED:
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, SELECTED_SUBDIR,
                                     f"{now_str()}_sel_f{fnum}_id{tid}_{obj_track_dict[tid]['cls_name']}_{obj_track_dict[tid]['conf']:.2f}.jpg"),
                        sel_frame
                    )
                if DEBUG_SAVE and SAVE_CROP:
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, CROP_SUBDIR,
                                     f"{now_str()}_crop_f{fnum}_id{tid}_{obj_track_dict[tid]['cls_name']}_{obj_track_dict[tid]['conf']:.2f}.jpg"),
                        crop_img
                    )

                if send_once:
                    sent_id.append(tid)

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
