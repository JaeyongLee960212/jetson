from ultralytics import YOLO

import numpy as np
import cv2
import os
import uuid
import time
from datetime import datetime
from collections import deque

from utils.loader import LCD_LoadStreams, LoadImagesAndVideos
from utils.comm_utils_lcs import Comm
from utils.check_lcs import Check
from conf.config import MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT, LCS_CLS_DIR


# =========================
# Options
# =========================
ASPECT_MAX = 1.2                # width > height = 1.2
MODEL_CONF = 0.6                # YOLO Confidence
TRACK_CLASSES = [0, 1, 2]       # Classes that we use
BUF_LEN = 5                     # Buffer Frame
PATIENCE = 8                    # Send Trigger -> If there is no object in ~ frame after last detection, then send 
EVENT_COOLDOWN_FRAMES = 150     # Don't Send after Sending LCS Information
LANE_COOLDOWN_SECS = 10.0       # _1,_2 COOLDOWN

# Ignore ~~~ Frame after detect LCS
IGNORE_WINDOW_FRAMES = 200

REQUIRE_TWO_FOR_SEND = True     # If you want to send LCS when _1,_2 both exists, set True
DEBUG = True                    # If you want to debug, set True, If you don't want to debug, set False

EXCLUDE_CLASS_NAMES = {"LCS_ROAD_BROKEN"}

def gen_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def xywh_to_xyxy(xywh, W, H):
    cx, cy, w, h = map(float, xywh)
    x1 = int(cx - w / 2); y1 = int(cy - h / 2)
    x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2


def main():
    model = YOLO("/data/YOLOv8_LCS/LCS_best.pt")
    model.fuse()

    #위에꺼(실시간), 밑에꺼(자체테스용, 동영상)
    dataset = LCD_LoadStreams(sources="192.168.10.116", vid_stride=1, buffer=True)
    # dataset = LoadImagesAndVideos("/data/YOLOv8_LCS/LCS_one_x_one_ok.avi", vid_stride=1)

    ch = Check()

    # save_obs: [{'fnum': int, 'objs': [obj, ...]}, ...]
    # obj = {'id','cx','cls_idx','cls_name','conf','xywh'}
    save_obs = deque(maxlen=BUF_LEN)
    save_frames = deque(maxlen=BUF_LEN)

    fnum = 0
    no_obs = 0
    event_cooldown = 0
    send_trigger = False

    lane_last_sent_ts = {'_1': 0.0, '_2': 0.0}

    ignore_left = 0

    global com

    for source, imgs, bs, count in dataset:
        np_img = np.array(imgs[0], dtype=np.uint8)
        H, W = np_img.shape[:2]
        fnum += 1

        if event_cooldown > 0:
            event_cooldown -= 1
        if ignore_left > 0:
            ignore_left -= 1

        results = model.track(
            np_img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=MODEL_CONF,
            classes=TRACK_CLASSES,
            verbose=False
        )
        results[0].names = LCS_CLS_DIR

        obs = []
        total_cnt = 0
        kept_broken = kept_aspect = kept_check = 0 

        for r in results:
            r.names = LCS_CLS_DIR
            ids = r.boxes.id

            for i, (cls, conf, xywh) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh)):
                total_cnt += 1
                cls = int(cls)
                conf = float(conf)
                xywh = xywh.tolist()
                w, h = float(xywh[2]), float(xywh[3])

                # Exception(BROKEN) #추후 BROKEN도 보려면 주석처리
                cls_name = r.names[cls]
                if (cls_name in EXCLUDE_CLASS_NAMES) or ("BROKEN" in cls_name.upper()):
                    kept_broken += 1
                    continue

                # (1) 비율 필터
                if w > h * ASPECT_MAX:
                    kept_aspect += 1
                    continue

                # (2) Check() ROI/크기/신뢰도 필터
                if not ch.check_valid(cls, xywh, conf):
                    kept_check += 1
                    continue

                # (3) 좌우 정렬용 center_x
                cx = xywh[0]

                # (4) 트랙 아이디
                trk_id = int(ids[i].item()) if ids is not None else None
                obj_id = f"trk_{trk_id}" if trk_id is not None else f"f{fnum}_i{i}_c{cls}"

                obs.append({
                    'id': obj_id,
                    'cx': cx,
                    'cls_idx': cls,
                    'cls_name': cls_name,
                    'conf': conf,
                    'xywh': xywh
                })

        if DEBUG:
            kept_cnt = len(obs)
            print(f"[F{fnum}] total={total_cnt} / kept={kept_cnt} (broken:{kept_broken}, aspect:{kept_aspect}, check:{kept_check})")

        if ignore_left > 0 or event_cooldown > 0:
            if save_obs or save_frames:
                save_obs.clear()
                save_frames.clear()
            continue

        # 프레임 기준 2개 좌,우 정렬 후 LCS_ROAD_USABLE_1,_2 순으로 순서별로 정렬
        frame_pack = {'fnum': fnum, 'objs': []}
        if len(obs) > 0:
            obs.sort(key=lambda o: o['conf'], reverse=True)
            obs = obs[:2]  # 상위 2개
            obs.sort(key=lambda o: o['cx'])  # 좌→우

            if len(obs) == 1:
                obs[0]['cls_name'] = f"{obs[0]['cls_name']}_1"
            elif len(obs) == 2:
                obs[0]['cls_name'] = f"{obs[0]['cls_name']}_1"
                obs[1]['cls_name'] = f"{obs[1]['cls_name']}_2"

            frame_pack['objs'] = obs

        if len(frame_pack['objs']) > 0:
            save_obs.append(frame_pack)
            save_frames.append(np_img.copy())

        if (len(obs) == 0) and (len(save_obs) > 0):
            no_obs += 1
            if no_obs > PATIENCE:
                send_trigger = True
        else:
            no_obs = 0

        if len(save_obs) == save_obs.maxlen:
            send_trigger = True

        if send_trigger:
            send_trigger = False
            event_cooldown = EVENT_COOLDOWN_FRAMES
            ignore_left = IGNORE_WINDOW_FRAMES 
            no_obs = 0 

            # 1) if LCS exists more than 2
            send_idx = None
            if REQUIRE_TWO_FOR_SEND:
                for i in range(len(save_obs) - 1, -1, -1):
                    if len(save_obs[i]['objs']) >= 2:
                        send_idx = i
                        break

            # case 2) if LCS exists less than 2, just send one
            if send_idx is None:
                for i in range(len(save_obs) - 1, -1, -1):
                    if len(save_obs[i]['objs']) >= 1:
                        send_idx = i
                        break
            if send_idx is None:
                save_obs.clear()
                save_frames.clear()
                continue

            send_pack = save_obs[send_idx]
            base_frame = save_frames[send_idx]

            if DEBUG:
                print(f"[SEND] choose frame fnum={send_pack['fnum']} objs={len(send_pack['objs'])}")

            now_ts = time.time()
            actually_sent = 0

            os.makedirs('data/test', exist_ok=True)

            for data in send_pack['objs']:
                cls_name = data.get('cls_name', 'unknown')
                suffix = '_1' if cls_name.endswith('_1') else ('_2' if cls_name.endswith('_2') else None)

                if suffix in lane_last_sent_ts:
                    elapsed = now_ts - lane_last_sent_ts[suffix]
                    if elapsed < LANE_COOLDOWN_SECS:
                        if DEBUG:
                            print(f"[COOLDOWN-SKIP] slot {suffix}: {elapsed:.1f}s < {LANE_COOLDOWN_SECS:.0f}s")
                        continue

                #Send
                detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                img_file_name = gen_uuid(detect_time) + '.jpg'

                msg = com.pack_data(
                    int(data.get('cls_idx', 0)),
                    cls_name,
                    float(data.get('conf', 0.0)),
                    data.get('xywh', [0, 0, 0, 0]),
                    send_pack['fnum'],
                    detect_time,
                    img_file_name
                )
                com.send_data_to_udp(msg)

                sel_frame = base_frame.copy()
                x1, y1, x2, y2 = xywh_to_xyxy(data['xywh'], base_frame.shape[1], base_frame.shape[0])
                cv2.rectangle(sel_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                com.upload_to_ftp(sel_frame, cls_name, img_file_name)

                safe_name = f"{cls_name}_{float(data.get('conf',0.0)):.2f}"
                with open(f"data/test/{safe_name}.txt", "w") as fw:
                    fw.write(msg)
                cv2.imwrite(f"data/test/{safe_name}.jpg", sel_frame)

                if DEBUG:
                    print("udp send ", msg)

                actually_sent += 1
                if suffix in lane_last_sent_ts:
                    lane_last_sent_ts[suffix] = now_ts

            if DEBUG and actually_sent == 0:
                print("[SEND] all candidates skipped by lane cooldown")

            save_obs.clear()
            save_frames.clear()


if __name__ == "__main__":
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()
