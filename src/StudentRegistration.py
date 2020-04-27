#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:59:29 2020

@author: jeevesh
"""


'''
Download : pip intall getkey
'''
from  camera import Webcam
import mysql.connector
from db.course_reg import Course_Reg
from mysql.connector import Error
from db.DBService import MyDatabase
import numpy as np
import cv2
from db.StudentLayer import Student
from getkey import getkey, keys
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#parent_dir = "/home/jeevesh/Desktop/SE/project/dataset/testingProject/"
parent_dir = "../data/train/"



class  StudentRegistration:
    
   # def __init__(self):
   
   
   
   
        
        
        
    def takeStudentDetails(self,rollNumber):
    
        firstName = input("Enter student First Name : or q to exit Registration " )
        if firstName == 'q':
            print("Registration Completed")
            exit()
            
        lastName =  input("Enter student Last Name : ")
        fullName = firstName + " " + lastName
        
        print("Entered Name is :")
        print(fullName)
        
        print("if Name is INcorrect and you want to re-enter the name press q/Q  ELSE press c/C to continue")
        key = getkey()
        if key == 'q' or key =='Q':
            return -1
        if key == 'c' or key =='C':
            print("We need to capture your image")
            print("press y/Y once you are ready")
            key = getkey()
            if key == 'y' or key == 'Y':
                frame = camera.takeStudentPhoto()
                #frame = self.takeStudentPhoto()
                self.createAugmentationAndSave(frame,rollNumber)
                #self.saveImages(frame,rollNumber)
        studentDict={}
        studentDict={'name':fullName,'facial_features':b'89'}
        return studentDict 
        
       
        
    def createAugmentationAndSave(self,image,rollNumber):
        directory = str(rollNumber)
        path = os.path.join(parent_dir, directory) 
        os.mkdir(path)
        
        folder_location = path + "/"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        image = img_to_array(image)  # this is a Numpy array with shape (3, 150, 150)
        x = image
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        photoCount = 0
        for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=folder_location, save_prefix=rollNumber, save_format='jpeg'):
            photoCount += 1
            if photoCount > 9:
                break  # otherwise the generator would loop indefinitely

        
        return
    
    
   
#ss=Course_Reg(infodict)
#ss.register_course()
    

    
print("Registration Started")
student = Student()
camera = Webcam()

print("press q , for closing the Registration")
while(True):
    
    studReg = StudentRegistration()
    nextRollNumber=student.getNextRollNumber()
    dataDict  = studReg.takeStudentDetails(nextRollNumber)
    sObject=Student(dataDict)
    sObject.createStudent()
    
    if 0xFF == ord('q'):
        exit()
    
print("Registration Completed")





'''
Following class : StudentRegistration deals with Registration of student at the time of admission.
takeStudentDetails  :  It take student details as input, if details are ok it calls to capture image of student
takeStudentPhoto   : it captures the student present in front of camera, this module present in camera.py
createAugmentationAndSave : it replicate different version of images and save in the path which is provided
'''