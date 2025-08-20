import cv2
import numpy as np
import matplotlib.pyplot as plt

# 영상 경로
video_path = "/data/YOLOv8_TL/Blink_test_1.mp4"

# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # 기본값

# 관심 영역 기준값
center_x, center_y = 554, 279
dx, dy = 5, 5
half_w, half_h = 15, 5  # width 30, height 10

brightness_list = []
frame_indices = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop box 계산
    x1 = center_x - dx - half_w
    y1 = center_y - dy - half_h
    x2 = center_x + dx + half_w
    y2 = center_y + dy + half_h

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    brightness_list.append(brightness)
    frame_indices.append(frame_idx)
    frame_idx += 1

cap.release()

# 시간 축 생성
t = np.linspace(0, len(brightness_list) / fps, len(brightness_list))

# FFT 계산
fft_result = np.fft.fft(brightness_list)
fft_freq = np.fft.fftfreq(len(brightness_list), d=1/fps)
amplitude = np.abs(fft_result)

# 양의 주파수만 시각화
pos_mask = fft_freq > 0
fft_freq = fft_freq[pos_mask]
amplitude = amplitude[pos_mask]

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, brightness_list)
plt.title("Brightness Over Time (ROI around (554, 279))")
plt.xlabel("Time (s)")
plt.ylabel("Brightness")

plt.subplot(1, 2, 2)
plt.plot(fft_freq, amplitude)
plt.title("Flicker Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 200)
plt.grid()

plt.tight_layout()
plt.show()
