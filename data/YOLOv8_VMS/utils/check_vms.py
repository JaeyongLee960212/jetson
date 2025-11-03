import numpy as np
from conf.config import LCD_CAMERA_IP, CAM_WIDTH, CAM_HEIGHT
import cv2

# ROI Check
TOP_ROI_START_X = 0
TOP_ROI_START_Y = 0
TOP_ROI_END_X = int(CAM_WIDTH * 2/3)
TOP_ROI_END_Y = int(CAM_HEIGHT * 1/3)


# Check Threshold : Set the Threshold Value that fits with your vehicle
class Check(): 
    def __init__(self):
        self.top_roi = [TOP_ROI_START_X, TOP_ROI_START_Y, TOP_ROI_END_X, TOP_ROI_END_Y]
        self.box_size_threshold = 12000 #14000 #10000
        #self.box_size_threshold = 85 * 355 
        self.confidence_threshold = 0.65 # 0.7
        self.count_threshold = 2 #2
        
        self.center_x = 0
        self.center_y = 0

        self.track_count = {}
        self.id_life = 0

    def check_count(self, id):
        if id not in self.track_count:
            self.track_count[id] = 1
        else:
            self.track_count[id] += 1

        return True if self.track_count[id] >= self.count_threshold else False

    def check_valid(self, cls, xywh, confidence) -> bool:
        # '''if use xyxy '''
        # self.center_x = xyxy[0]+((xyxy[2]-xyxy[0])/2)
        # self.center_y = xyxy[1]+((xyxy[3]-xyxy[1])/2)
        self.center_x = xywh[0]
        self.center_y = xywh[1]
        self.box_width = xywh[2]
        self.box_height = xywh[3]

    #0. Base Check. (all class apply)
        #Object location check (Left or Right)
        is_lr = "right" if self.center_x >= CAM_WIDTH/2 else "left"

        #ROI Check (edge roi box which left, right, top)
        top_roi_ch = True if (self.top_roi[1] < self.center_y <= self.top_roi[3]) else False

        #Box Size Check
        box_size_ch = True if (self.box_width * self.box_height >= self.box_size_threshold) else False

        conf_ch = True if(confidence >= self.confidence_threshold) else False
        
        ch = ((top_roi_ch and conf_ch) and box_size_ch)
        #is_lr=="left"

        if ch: return True
        else: return False 

    def draw_zone(img):
        img = cv2.rectangle(img,
                        (TOP_ROI_START_X, TOP_ROI_START_Y),
                        (TOP_ROI_END_X, TOP_ROI_END_Y),
                        (255,0,20), # bgr
                        2)
        return img

    def check_valid(self, cls, xywh, confidence):
        
        # self.center_x = xyxy[0]+((xyxy[2]-xyxy[0])/2)
        # self.center_y = xyxy[1]+((xyxy[3]-xyxy[1])/2)
        self.center_x = xywh[0]
        self.center_y = xywh[1]
        self.box_width = xywh[2]
        self.box_height = xywh[3]

        #TODO ROI Check (xy)
        # self.left_roi_ch = True if (self.left_roi[0] < self.center_x <= self.left_roi[2]) else False
        # self.right_roi_ch = True if (self.right_roi[0] <= self.center_x < self.right_roi[2]) else False
        # self.top_roi_ch = True if (self.top_roi[1] < self.center_y <= self.top_roi[3]) else False

        #Checked Left Line
        if( (self.left_roi[0] < self.center_x <= self.left_roi[2])): # and (Left_roi[1] < center_y <= Left_roi[3])
            self.left_roi_ch = True
            print("l_check")
            return True
        else : self.left_roi_ch = False
        
        #Checked Right Line
        if(self.right_roi[0] <= self.center_x < self.right_roi[2]): # and (right_roi[1] < center_y <= right_roi[3])
            self.right_roi_ch = True
            print("r_check")
            return True
        else : self.right_roi_ch = False

        #Checked Center Line
        if(self.top_roi[1] < self.center_y <= self.top_roi[3]):
            self.top_roi_ch = True
            print("top_check")
            return True
        else : self.top_roi_ch = False
        
        #box size check
        w = xywh[2]
        h = xywh[3]
        box_size = w*h

        return False
