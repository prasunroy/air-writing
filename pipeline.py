# -*- coding: utf-8 -*-
"""
Air-Writing pipeline.
Created on Sat May 12 20:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/air-writing

"""


# imports
from __future__ import division

import cv2
import numpy

from recognizer import Recognizer


# Pipeline class
class Pipeline(object):
    
    # ~~~~~~~~ constructor ~~~~~~~~
    def __init__(self):
        # lower and upper bound for marker color
        self._lower_hue_0 = numpy.array([80, 90, 100])
        self._lower_hue_1 = numpy.array([120, 255, 255])
        self._upper_hue_0 = numpy.array([80, 90, 100])
        self._upper_hue_1 = numpy.array([120, 255, 255])
        
        # image processing kernels
        self._kernel_median_blur = 27
        self._kernel_dilate_mask = (9, 9)
        
        # marker properties
        self._x = -1
        self._y = -1
        self._dx = 0
        self._dy = 0
        self._vx = 0
        self._vy = 0
        self._histdx = []
        self._histdy = []
        self._points = []
        self._max_points = 50
        self._min_change = 10
        self._min_veloxy = 2.0
        self._marker_ctr = None
        self._marker_tip = None
        
        # frames per second
        self._fps = 20
        
        # render elements
        self._render_marker = True
        self._render_trails = True
        
        # recognizer
        self._recognizer = Recognizer()
        
        # opencv version
        self._opencv_version = int(cv2.__version__.split('.')[0])
        
        return
    
    # ~~~~~~~~ marker segmentation ~~~~~~~~
    def _marker_segmentation(self, frame):
        # convert RGB to HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # create mask for marker
        mask_0 = cv2.inRange(frame_hsv, self._lower_hue_0, self._lower_hue_1)
        mask_1 = cv2.inRange(frame_hsv, self._upper_hue_0, self._upper_hue_1)
        
        mask = cv2.addWeighted(mask_0, 1.0, mask_1, 1.0, 0.0)
        
        # remove noise from mask
        mask = cv2.medianBlur(mask, self._kernel_median_blur)
        
        # perform dilation on mask
        mask = cv2.dilate(mask, self._kernel_dilate_mask)
        
        return mask
    
    # ~~~~~~~~ marker tip identification ~~~~~~~~
    def _marker_tip_identification(self, mask):
        # find contours in mask
        if self._opencv_version == 2:
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        else:
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        
        # process contours
        if contours and len(contours) > 0:
            # find the largest contour
            contour_max = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
            # find tip of marker assuming as peak of largest contour
            contour_roi = contour_max.reshape(contour_max.shape[0], contour_max.shape[2])
            contour_roi = sorted(contour_roi, key=lambda x:x[1])
            
            marker_tip = (contour_roi[0][0], contour_roi[0][1])
        else:
            contour_max = None
            marker_tip = None
        
        return [contour_max, marker_tip]
    
    # ~~~~~~~~ trajectory approximation ~~~~~~~~
    def _trajectory_approximation(self, marker_tip, frame):
        image = None
        if marker_tip is None:
            # reset marker
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
        else:
            # check buffers
            if len(self._histdx) >= self._fps:
                self._histdx.pop(0)
            if len(self._histdy) >= self._fps:
                self._histdy.pop(0)
            if len(self._points) > self._max_points:
                self._points.pop(0)
            
            # update position and velocity of marker
            if self._x < 0 or self._y < 0:
                self._x, self._y = marker_tip
            self._dx = abs(marker_tip[0] - self._x)
            self._dy = abs(marker_tip[1] - self._y)
            self._histdx.append(self._dx)
            self._histdy.append(self._dy)
            if self._dx > self._min_change or self._dy > self._min_change:
                self._points.append(marker_tip)
            self._x, self._y = marker_tip
            
            self._vx = numpy.floor(sum(self._histdx[-self._fps:]) / self._fps)
            self._vy = numpy.floor(sum(self._histdy[-self._fps:]) / self._fps)
            
            # draw trajectory of marker if marker is in static state
            nodes = len(self._points)
            if nodes > 1:
                image = numpy.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
                for i in range(nodes-1):
                    cv2.line(image, self._points[i], self._points[i+1], (255, 255, 255), 4, cv2.LINE_AA)
        
        return image
    
    # ~~~~~~~~ character recognition ~~~~~~~~
    def _character_recognition(self, image, engine):
        predictions = self._recognizer.predict(image, engine)
        
        return predictions
    
    # ~~~~~~~~ render frame ~~~~~~~~
    def _render(self, frame):
        if not self._marker_ctr is None:
            cv2.drawContours(frame, self._marker_ctr, -1, (0, 255, 0), 1)
        if not self._marker_tip is None:
            cv2.circle(frame, self._marker_tip, 4, (255, 255, 0), -1)
        n = len(self._points)
        if n > 1:
            for i in range(n-1):
                cv2.line(frame, self._points[i], self._points[i+1], (255, 255, 0), 4, cv2.LINE_AA)
        
        return frame
    
    # ~~~~~~~~ run inference ~~~~~~~~
    def run_inference(self, frame, engine='EN'):
        # STEP-A: marker segmentation
        mask = self._marker_segmentation(frame)
        
        # STEP-B: marker tip identification
        self._marker_ctr, self._marker_tip = self._marker_tip_identification(mask)
        
        # STEP-C: trajectory approximation
        image = self._trajectory_approximation(self._marker_tip, frame)
        
        # STEP-D: character recognition
        if not image is None and self._vx < self._min_veloxy and self._vy < self._min_veloxy:
            prediction, predprobas = self._character_recognition(image, engine)
            
            # reset marker
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
            self._marker_ctr = None
            self._marker_tip = None
        else:
            prediction, predprobas = [None, None]
        
        # render frame
        frame = self._render(frame)
        
        return [prediction, predprobas, mask, frame]
