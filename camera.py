# -*- coding: utf-8 -*-
"""
Interface for camera hardware.
Created on Fri May 11 22:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/air-writing

"""


# imports
import cv2


# VideoStream class
class VideoStream(object):
    
    # ~~~~~~~~ constructor ~~~~~~~~
    def __init__(self, src=0):
        self.video = cv2.VideoCapture(src)
        
        return
    
    # ~~~~~~~~ set target frame dimension ~~~~~~~~
    def setFrameSize(self, size=(-1, -1)):
        if size[0] > 0:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        if size[1] > 0:
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        
        return
    
    # ~~~~~~~~ get frame from device ~~~~~~~~
    def getFrame(self, flip=None):
        self.frame = self.video.read()[1]
        if not self.frame is None:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if type(flip) is int:
                self.frame = cv2.flip(self.frame, flip)
        
        return self.frame
    
    # ~~~~~~~~ clean up and release resources ~~~~~~~~
    def clear(self):
        self.video.release()
        
        return
