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
# 파라미터 / 옵션
# =========================
ASPECT_MAX = 1.2                # w > h*ASPECT_MAX 면 제외
MODEL_CONF = 0.6                # YOLO confidence
TRACK_CLASSES = [0, 1, 2]       # 사용할 클래스 인덱스
BUF_LEN = 5                     # 버퍼 프레임 수
PATIENCE = 8                    # 관측 없음 지속 프레임 → 전송 트리거
EVENT_COOLDOWN_FRAMES = 150     # 전역(이벤트) 쿨다운(프레임)
LANE_COOLDOWN_SECS = 10.0       # 슬롯별(_1/_2) 쿨다운(초)

# ⭐ 전송 직후, 이 프레임 수 만큼은 감지·버퍼링 자체를 하드 차단
IGNORE_WINDOW_FRAMES = 200

REQUIRE_TWO_FOR_SEND = True     # True면 _1, _2 둘 다 있을 때만 전송
DEBUG = True                    # 디버그 출력

# broken(파손) 제외
EXCLUDE_CLASS_NAMES = {"LCS_ROAD_BROKEN"}


# =========================
# 유틸
# =========================
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

    # 입력 소스
    dataset = LCD_LoadStreams(sources="192.168.10.116", vid_stride=1, buffer=True)
    # dataset = LoadImagesAndVideos("/data/YOLOv8_LCS/LCS_one_x_one_ok.avi", vid_stride=1)

    ch = Check()

    # 버퍼
    # save_obs: [{'fnum': int, 'objs': [obj, ...]}, ...]
    # obj = {'id','cx','cls_idx','cls_name','conf','xywh'}
    save_obs = deque(maxlen=BUF_LEN)
    save_frames = deque(maxlen=BUF_LEN)

    fnum = 0
    no_obs = 0
    event_cooldown = 0
    send_trigger = False

    # 슬롯별 마지막 전송 시각
    lane_last_sent_ts = {'_1': 0.0, '_2': 0.0}

    # ⭐ 전송 직후 하드 무시창 카운터
    ignore_left = 0

    global com

    for source, imgs, bs, count in dataset:
        np_img = np.array(imgs[0], dtype=np.uint8)
        H, W = np_img.shape[:2]
        fnum += 1

        # 쿨다운/하드무시창 감소
        if event_cooldown > 0:
            event_cooldown -= 1
        if ignore_left > 0:
            ignore_left -= 1

        # 모델 추론 (fps 유지 목적으로는 계속 돌리되, 이후 로직에서 하드 차단 적용)
        results = model.track(
            np_img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=MODEL_CONF,
            classes=TRACK_CLASSES,
            verbose=False
        )
        results[0].names = LCS_CLS_DIR

        # ───── 후보 수집 & 필터 ─────
        obs = []
        total_cnt = 0
        kept_broken = kept_aspect = kept_check = 0  # 디버그 카운터

        for r in results:
            r.names = LCS_CLS_DIR
            ids = r.boxes.id

            for i, (cls, conf, xywh) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh)):
                total_cnt += 1
                cls = int(cls)
                conf = float(conf)
                xywh = xywh.tolist()
                w, h = float(xywh[2]), float(xywh[3])

                # (0) broken 제외
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

        # ───── 하드 무시창/전역 쿨다운 동안은 버퍼·트리거 완전 차단 ─────
        if ignore_left > 0 or event_cooldown > 0:
            # 버퍼에 뭔가 남아 있으면 비움 (특히 전송 직후 다음 프레임 등에 남는 것 방지)
            if save_obs or save_frames:
                save_obs.clear()
                save_frames.clear()
            # 이 프레임은 완전 무시 (no_obs/트리거 갱신도 안 함)
            continue

        # ───── 프레임 내 상위 2개 선택 → 좌→우 정렬 → _1/_2 라벨 ─────
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

        # 버퍼 저장
        if len(frame_pack['objs']) > 0:
            save_obs.append(frame_pack)
            save_frames.append(np_img.copy())

        # ───── 전송 트리거 ─────
        if (len(obs) == 0) and (len(save_obs) > 0):
            no_obs += 1
            if no_obs > PATIENCE:
                send_trigger = True
        else:
            no_obs = 0

        if len(save_obs) == save_obs.maxlen:
            send_trigger = True

        # ───── 전송 ─────
        if send_trigger:
            send_trigger = False
            event_cooldown = EVENT_COOLDOWN_FRAMES
            ignore_left = IGNORE_WINDOW_FRAMES   # ⭐ 전송 직후 하드 무시창 가동
            no_obs = 0  # 전송 후 초기화

            # 1) 버퍼 뒤에서부터 2개짜리 프레임 찾기
            send_idx = None
            if REQUIRE_TWO_FOR_SEND:
                for i in range(len(save_obs) - 1, -1, -1):
                    if len(save_obs[i]['objs']) >= 2:
                        send_idx = i
                        break

            # 2) 둘 다 없으면 1개라도
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

                # 슬롯별 쿨다운 체크
                if suffix in lane_last_sent_ts:
                    elapsed = now_ts - lane_last_sent_ts[suffix]
                    if elapsed < LANE_COOLDOWN_SECS:
                        if DEBUG:
                            print(f"[COOLDOWN-SKIP] slot {suffix}: {elapsed:.1f}s < {LANE_COOLDOWN_SECS:.0f}s")
                        continue

                # 전송
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

            # 다음 이벤트 대비 버퍼 비움 (특히 하드무시창 시작 직전에 반드시 비움)
            save_obs.clear()
            save_frames.clear()


if __name__ == "__main__":
    com = Comm(MAIN_SERVER_IP, MAIN_SERVER_UDP_PORT)
    main()
