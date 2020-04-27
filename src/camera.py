#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:38:49 2020

@author: jeevesh
"""
import os
import cv2
    
class Webcam:
    
    def takeStudentPhoto(self):
        frame = None
        cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
        ret,frame = cap.read() # return a single frame in variable `frame`
        while(True):
            cv2.imshow('img1',frame) #display the captured image
            if cv2.waitKey(1) & 0xFF == ord('q'): #save on pressing 'y' 
                print("photo save successfully")
                cv2.destroyAllWindows()
                break
        return frame
    
    
    
    
    
    
'''
takeStudentPhoto: It capture the photo, and return the frame along with this
it shows the photo in small window
If you want to close the window , press 'q'

'''