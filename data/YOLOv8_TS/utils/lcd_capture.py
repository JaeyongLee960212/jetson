from arena_api.system import system
from arena_api.buffer import *

import ctypes
import numpy as np
import time

"""
demonstrates live stream
(1) Start device stream
(2) Get a buffer and create a copy
(3) Requeue the buffer
(4) Calculate bytes per pixel for reshaping
(5) Create array from buffer cpointer data
(6) Create a NumPy array with the image shape
(7) Display the NumPy array using OpenCV
(8) When Esc is pressed, stop stream and destroy OpenCV windows
"""
TAB1 = "  "
TAB2 = "    "
NUM_SECONDS = 10

# Image timeout
TIMEOUT_MILLISEC = 2000

class Stream_Vision_Cam:
    def __init__(self):
        self.pixelformat = 'BGR8'
        self.curr_frame_time = 0
        self.prev_frame_time = 0      
        self.fps = 0.0  
        self.fc = 0                 #frame Count
        
    def terminate_stream(self):
        self.device.stop_stream()   #End Stream
        system.destroy_device()     #Clear Device
        print("end stream \nclear cam device")
    
    def is_Opened(self):
        return True if self.device is not None else False

    def create_devices_with_tries(self):
            print("search CAM")
            time.sleep(10)
            tries = 0
            tries_max = 6
            sleep_time_secs = 10
            # print("device info : ",system.device_infos)
            while tries < tries_max:  # Wait for device for 60 seconds
                devices = system.create_device()
                if not devices:
                    print(
                        f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                        f'secs for a device to be connected!')
                    for sec_count in range(sleep_time_secs):
                        time.sleep(1)
                        print(f'{sec_count + 1 } seconds passed ',
                            '.' * sec_count, end='\r')
                    tries += 1
                    
                else:
                    print(f'Created {len(devices)} device(s)')
                    return devices
            else:
                raise Exception(f'No device found! Please connect a device and run '
                                f'the example again.')
    
    def setup(self, cam_ip='192.168.10.116', width=1280, height=720):
        
        self.width = width
        self.height = height
        self.cam_ip = cam_ip
        print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
        print('\nCam Setup started... \n')

        self.devices = self.create_devices_with_tries()     # waits for the user to connect a device before raising
    
        #find 2 camera   
        for d in self.devices:
            if (str(d).find(self.cam_ip) != -1):
                self.device = d
                print("device :",self.device)
                break
            else:
                self.device = None
        self.device = self.devices[0]
        
        print(f'{TAB1}Enable multicast')
        self.device.tl_stream_nodemap['StreamMulticastEnable'].value = True
        self.device_access_status = self.device.tl_device_nodemap['DeviceAccessStatus'].value
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",self.device_access_status)
        
        self.num_channels = 6
        
        # Master
        if self.device_access_status == 'ReadWrite':

            print(f'{TAB1}Host streaming as "master"')

            # Get node values that will be changed in order to return their values
            # at the end of the example
            acquisition_mode_initial = self.device.nodemap['AcquisitionMode'].value

            # Set acquisition mode
            print(f'{TAB1}Set acquisition mode to "Continuous"')

            self.device.nodemap['AcquisitionMode'].value = 'Continuous'

            # Enable stream auto negotiate packet size
            self.device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

            # Enable stream packet resend
            self.device.tl_stream_nodemap['StreamPacketResendEnable'].value = True

            self.nodemap = self.device.nodemap
            self.nodes = self.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
            self.nodes['Width'].value = self.width
            self.nodes['Height'].value = self.height
            self.nodes['PixelFormat'].value = self.pixelformat

            # Stream nodemap
            self.device.tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
            self.device.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
            self.device.tl_stream_nodemap['StreamPacketResendEnable'].value = True
            
        # Listener
        else:
            print(f'{TAB1}Host streaming as "listener"\n')

        # Get images
        print(f'{TAB1}Getting images for {NUM_SECONDS} seconds')

        print('\nStart Stream... \n')
        self.device.start_stream() # Start Stream
        self.capture() # curr_frame_time
        print("test capture")
    
    # Due to yolov8 LoadStream structure, one frame should be taken...
    def capture(self):
        #with self.device.start_stream():      
        self.curr_frame_time = time.time()     # Used to display FPS on stream
        self.buffer = self.device.get_buffer() # Copy buffer and requeue to avoid running out of buffers
        self.item = BufferFactory.copy(self.buffer)
        self.device.requeue_buffer(self.buffer)
        self.buffer_bytes_per_pixel = int(len(self.item.data)/(self.item.width * self.item.height))
        self.array = (ctypes.c_ubyte * self.num_channels * self.item.width * self.item.height).from_address(ctypes.addressof(self.item.pbytes)) # Buffer data as cpointers can be accessed using buffer.pbytes
        self.npndarray = np.ndarray(buffer=self.array, dtype=np.uint8, shape=(self.item.height, self.item.width, self.buffer_bytes_per_pixel)) # Create a reshaped NumPy array to display using OpenCV
        self.fps = round(1/(self.curr_frame_time - self.prev_frame_time),6)
        self.fc += 1
        BufferFactory.destroy(self.item) # Destroy the copied item to prevent memory leaks
        self.prev_frame_time = self.curr_frame_time
        #print("capture fps",self.fps)
        return True, self.npndarray

