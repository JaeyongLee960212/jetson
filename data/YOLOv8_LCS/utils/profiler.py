import time
import contextlib
import torch

class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
    

"""
    example) 

    dt = (Profile(), Profile(), Profile(), Profile())

    with dt[0]:
        ...
    with dt[1]:
        ...
    with dt[2]:
        ...
    with dt[3]:
        ...

    print(f"전체 실행 시간 : {dt[0].t}, \n\
        {dt[1].t} \n\
        {dt[2].t} \n\
        {dt[3].t}, \n\
        {dt[4].t}")
    
    print(f"dataset load : {(dt[1].t/(dt[0].t))*100}% \n\
        {(dt[2].t/(dt[0].t))*100}% \n\
        {(dt[3].t/(dt[0].t))*100}% \n\
        {(dt[4].t/(dt[0].t))*100}%")
        
"""