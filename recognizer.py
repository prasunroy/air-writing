# -*- coding: utf-8 -*-
"""
Optical character recognition in air-writing.
Created on Sun May 13 20:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/air-writing

"""


# imports
from __future__ import division

# ---- compatibility for MKL 2018 ----
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

# ---- main modules ----
import cv2
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend


# setup backend
backend.set_image_dim_ordering('th')


# Recognizer class
class Recognizer(object):
    
    # ~~~~~~~~ constructor ~~~~~~~~
    def __init__(self):
        # model properties
        self._i_shape = (1, 56, 56)
        self._b_shape = (1, 40, 40)
        self._n_class = 10
        self._path_en = 'models/en_numbers_ft.h5'
        self._path_bn = 'models/bn_numbers_ft.h5'
        self._path_dv = 'models/dv_numbers_ft.h5'
        self._model_en = self._cnn(self._i_shape, self._n_class, self._path_en)
        self._model_bn = self._cnn(self._i_shape, self._n_class, self._path_bn)
        self._model_dv = self._cnn(self._i_shape, self._n_class, self._path_dv)
        
        # image processing
        self._min_size = 8
        self._d_kernel = (3, 3)
        
        # opencv version
        self._opencv_version = int(cv2.__version__.split('.')[0])
        
        return
    
    # ~~~~~~~~ CNN architecture ~~~~~~~~
    def _cnn(self, i_shape=(1, 28, 28), n_class=10, weights=None):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=i_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=n_class, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        if not weights is None and os.path.isfile(weights):
            model.load_weights(weights)
        
        return model
    
    # ~~~~~~~~ resize image ~~~~~~~~
    def _resize(self, image):
        w = image.shape[1]
        h = image.shape[0]
        dst_w = self._i_shape[1]
        dst_h = self._i_shape[2]
        box_w = self._b_shape[1]
        box_h = self._b_shape[2]
        
        if w >= h:
            new_h = h * box_w // w
            image = cv2.resize(image, (box_w, new_h), interpolation=cv2.INTER_AREA)
            pad_w = (dst_w - box_w) // 2
            pad_h = (dst_h - new_h) // 2
            pad_l = numpy.zeros((new_h, pad_w), dtype='uint8')
            pad_r = numpy.zeros((new_h, pad_w), dtype='uint8')
            pad_t = numpy.zeros((pad_h, dst_w), dtype='uint8')
            pad_b = numpy.zeros((dst_h-new_h-pad_h, dst_w), dtype='uint8')
            image = numpy.hstack((pad_l, image, pad_r))
            image = numpy.vstack((pad_t, image, pad_b))
        else:
            new_w = w * box_h // h
            image = cv2.resize(image, (new_w, box_h), interpolation=cv2.INTER_AREA)
            pad_w = (dst_w - new_w) // 2
            pad_h = (dst_h - box_h) // 2
            pad_l = numpy.zeros((box_h, pad_w), dtype='uint8')
            pad_r = numpy.zeros((box_h, dst_w-new_w-pad_w), dtype='uint8')
            pad_t = numpy.zeros((pad_h, dst_w), dtype='uint8')
            pad_b = numpy.zeros((pad_h, dst_w), dtype='uint8')
            image = numpy.hstack((pad_l, image, pad_r))
            image = numpy.vstack((pad_t, image, pad_b))
        
        return image
    
    # ~~~~~~~~ predict ~~~~~~~~
    def predict(self, image, engine='EN'):
        # find contours in image
        if self._opencv_version == 2:
            contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        else:
            contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        
        # find bounding rectangle around each contour
        bn_rects = []
        
        for cntr in contours:
            bn_rects.append(cv2.boundingRect(cntr))
        
        # sort bounding rectangles from left to right
        bn_rects.sort(key=lambda x: x[0])
        
        # process each bounding rectangle
        prediction = []
        predprobas = []
        
        for rect in bn_rects:
            # attributes of bounding rectangle
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            
            # ignore tiny objects assuming them as noise
            if h < self._min_size:
                continue
            
            # extract region of interest
            image = image[y:y+h, x:x+w]
            
            # resize region of interest
            image = self._resize(image)
            
            # perform dilation
            image = cv2.dilate(image, self._d_kernel)
            
            # reshape and scale features
            image = image.astype('float64').reshape(1, 1, self._i_shape[2], self._i_shape[1]) / 255.0
            
            # predict label
            if engine.upper() == 'EN':
                prob = self._model_en.predict(image)
            elif engine.upper() == 'BN':
                prob = self._model_bn.predict(image)
            elif engine.upper() == 'DV':
                prob = self._model_dv.predict(image)
            
            pred = numpy.argmax(prob)
            prob = numpy.round(numpy.max(prob), 4)
            
            # append result to list
            prediction.append(str(pred))
            predprobas.append(str(prob))
        
        return [prediction, predprobas]
