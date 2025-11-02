import numpy as np
from conf.config import LCD_CAMERA_IP, CAM_WIDTH, CAM_HEIGHT
import cv2

TOP_ROI_START_X = int(CAM_WIDTH * 1/5)
TOP_ROI_START_Y = 0
TOP_ROI_END_X = CAM_WIDTH #int(CAM_WIDTH * 6/7) 
TOP_ROI_END_Y = int(CAM_HEIGHT * 1.7/5) #int(CAM_HEIGHT * 1/5)


#class Check(left_roi:list, right_roi:list, top_roi:list, box_size_th:int, conf_th:float): 
class Check(): 
    def __init__(self):
        self.top_roi = [TOP_ROI_START_X, TOP_ROI_START_Y, TOP_ROI_END_X, TOP_ROI_END_Y]
        self.box_size_threshold = 500 #1400 #500
        # self.box_size_threshold = 85 * 355 
        self.confidence_threshold = 0.6 #0.6
        self.count_threshold = 3 #3
        
        self.center_x = 0
        self.center_y = 0

        self.track_count = {}
        self.id_life = 0
        self.cls_count = {}
        self.check_LCS ={}
    
    def check_count(self, id, id_cls):
        
        if id_cls not in self.track_count:
            self.track_count[id_cls] = 1
        else:
            self.track_count[id_cls] += 1

        return True if self.track_count[id_cls] >= self.count_threshold else False

    def check_valid(self, cls, xywh, confidence) -> bool:
        # '''if use xyxy '''
        # self.center_x = xyxy[0]+((xyxy[2]-xyxy[0])/2)
        # self.center_y = xyxy[1]+((xyxy[3]-xyxy[1])/2)
        self.center_x = xywh[0]
        self.center_y = xywh[1]
        self.box_width = xywh[2]
        self.box_height = xywh[3]

        if self.box_height > self.box_width:
            return False
        
    #0. Base Check. (all class apply)
        #Object location check (Left or Right)
        # is_lr = "ON_LANE2" if self.center_x >= 950 else "ON_LANE1"

        #ROI Check (edge roi box which left, right, top)
        top_roi_ch = True if (self.top_roi[1] < self.center_y <= self.top_roi[3]) and (self.top_roi[0] < self.center_x <= self.top_roi[2]) else False
            
        #Box Size Check
        box_size_ch = True if (self.box_width * self.box_height >= self.box_size_threshold) else False

        conf_ch = True if(confidence >= self.confidence_threshold) else False

    #1. Traffic Sign Case
    #2. Signal Case

        if int(cls) == 4: # yellow
            return top_roi_ch and conf_ch
    
        ch = ((top_roi_ch) and \
            # is_lr=="right" or 
            box_size_ch and \
            conf_ch)
        
        if ch: return True
        else: return False


    #Confidence Check 
    # conf_check = True if confidence >= 0.7 else False

    # if box_size >= 1000 and conf_check:
    #     print(f"confidence : {confidence} box size : {box_size} Send Message")
    #     return True
    # else:
    #     print(f"confidence : {confidence} box size : {box_size} Don't Send Message")
    #     return False

    # cv2.imshow("Tracking", annotated_frame)
    #1280 * 720. 
    # return True if (w*h > 400) else False
    ...


    def draw_zone(self, img):
        img = cv2.rectangle(img,
                            (TOP_ROI_START_X, TOP_ROI_START_Y),
                            (TOP_ROI_END_X, TOP_ROI_END_Y),
                            (255,0,20), # bgr
                            2)
        return img


# 곡선을 정의하는 점들
# points = np.array([[0, height], [width//2, -height], [width, height]])
# t_values = np.linspace(0, 1, 100)
# curve_points = np.array([((1-t)**2)*points[0] + (2*(1-t)*t)*points[1] + (t**2)*points[2] for t in t_values], np.int32)
# 곡선 그리기
# cv2.polylines(annotated_frame, [curve_points], isClosed=False, color=(255, 255, 255), thickness=2)
'''

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

'''
"""
if __name__ == "__main__":
    img_path = "/home/traffic/vms_ocr/frame_0100.jpg"
    img = cv2.imread(img_path)
    
    right_roi = [RIGHT_ROI_START_X, RIGHT_ROI_START_Y, RIGHT_ROI_END_X, RIGHT_ROI_END_Y]
    top_roi = [TOP_ROI_START_X, TOP_ROI_START_Y, TOP_ROI_END_X, TOP_ROI_END_Y]
    
    img = cv2.rectangle(img,
                          (RIGHT_ROI_START_X, RIGHT_ROI_START_Y),
                          (RIGHT_ROI_END_X, RIGHT_ROI_END_Y),
                          (0,0,255), # bgr
                          2)
    
    img = cv2.rectangle(img,
                          (TOP_ROI_START_X, TOP_ROI_START_Y),
                          (TOP_ROI_END_X, TOP_ROI_END_Y),
                          (255,0,20), # bgr
                          2)
    
    cv2.imshow(img)
"""
