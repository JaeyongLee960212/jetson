import math
import os
import time
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
from ultralytics.utils import LOGGER, ops

from conf.config import LCD_CAMERA_IP, CAM_WIDTH, CAM_HEIGHT, FPS
from utils.lcd_capture import Stream_Vision_Cam


class LCD_LoadStreams:

    def __init__(self, sources=LCD_CAMERA_IP, vid_stride=1, buffer=False):
        
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.caps = [0] * n
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.count = 0

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam

            if isinstance(sources, (int)):
                self.caps = [None] * n  # usb camera 
            else: 
                self.caps[i] = Stream_Vision_Cam()  # store video capture object
                self.caps[i].setup(cam_ip= LCD_CAMERA_IP, width=CAM_WIDTH, height=CAM_HEIGHT)
                
            if not self.caps[i].is_Opened():
                raise ConnectionError(f"{st}Failed to open {s}")
            
            w = int(self.caps[i].width)
            h = int(self.caps[i].height)

            fps = self.caps[i].fps  # warning: may return 0 or nan
            print("cam fps :", fps)
            # self.frames[i] = max(int(self.caps[i].fc), 0) or float("inf")  # infinite stream fallback
            self.frames[i] = float("inf")  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
            success, im = self.caps[i].capture()  # guarantee first frame                 self.caps[i].read 

            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()

        LOGGER.info("")  # newline

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        
        while self.running and cap.is_Opened() and n < (f):
            # print(self.threads)
            if len(self.imgs[i]) < 10:  # keep a <=10-image buffer
                n += 1
                #cap.grab() #usb camera 
                if n % self.vid_stride == 0:
                    #success, im = cap.retrieve() #usb camera 
                    success, im = cap.capture()
                    
                    # im = im[::2, ::2, :] # resize images
                    # print("IMSHAPE:", im.shape)
                    #update test cv2 imshow
                    # np_img = np.array(im, dtype=np.uint8)
                    # cv2.imshow("Tracking", np_img)
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break
                    #####
                    #Modi - FPS 리스트를 스칼라로 덮어쓰는 레이스 버그 -> time.sleep(1/self.fps) 
                    # 갱신
                    fps_val = cap.fps if (cap.fps and cap.fps > 0) else 30
                    self.fps[i] = fps_val

                    # 대기
                    delay = 1.0 / max(1.0, self.fps[i])  # 하한 1 FPS
                    time.sleep(min(delay, 0.033))         # 상한도 33ms 정도로 캡

                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        print("cloase")
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.terminate_stream()  # release video capture
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")

        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns source paths, transformed and original images for processing."""
        self.count += 1
        images = []
        for i, x in enumerate(self.imgs):
            
            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive():  # q to quit
                    self.close()
                    raise StopIteration
                
                time.sleep(1 / self.fps) # wait 1 / update fps 
                # time.sleep(0.05)

                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))
                #img = x.pop(0)

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs, self.count

    def __len__(self):
        """Return the length of the sources object."""
        return self.bs  # 1E12 frames = 32 streams at 30 FPS for 30 years


import glob
from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS

class LoadImagesAndVideos:
    """
    YOLOv8 image/video dataloader.

    This class manages the loading and pre-processing of image and video data for YOLOv8. It supports loading from
    various formats, including single image files, video files, and lists of image and video paths.

    Attributes:
        files (list): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (list): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate, defaults to 1.
        bs (int): Batch size, set to 1 for this class.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _new_video(path): Create a new cv2.VideoCapture object for a given video path.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images or videos
        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames along with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if len(imgs) > 0:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = cv2.imread(path)  # BGR
                if im0 is None:
                    raise FileNotFoundError(f"Image Not Found {path}")
                paths.append(path)
                imgs.append(im0)
                info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info, self.count

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(self.nf / self.bs)  # number of files

