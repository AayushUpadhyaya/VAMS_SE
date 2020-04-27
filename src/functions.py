#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following program contains the fuctions which we are going 
to use in our main file 


"""


from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os
from os.path import isdir
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import pickle



# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
  image = Image.open(filename)
  image = image.convert('RGB')
  pixels = asarray(image)
  detector = MTCNN()
  results = detector.detect_faces(pixels)
  if len(results)==0 :
    return []
  x1, y1, width, height = results[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array




# load images and extract faces for all images in a directory
def load_faces(directory):
  faces = list()
  for filename in listdir(directory):
    path = directory + filename
    face = extract_face(path)
    if len(face) == 0:
      continue
    faces.append(face)
  return faces



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
  X, y = list(), list()
  for subdir in listdir(directory):
    path = directory + subdir + '/'
    if not isdir(path):
      continue
    faces = load_faces(path)
    labels = [subdir for _ in range(len(faces))]
    X.extend(faces)
    y.extend(labels)
  return asarray(X), asarray(y)



# get the face embedding for one face
def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]


# extract a single face from a given photograph
def extract_face_webcam(image, required_size=(160, 160)):
    #image = Image.open(filename)
    #image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results)==0 :
        return []
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array,x1,y1,x2,y2








def try_extract_face_webcam(image, required_size=(160, 160)):
    #image = Image.open(filename)
    #image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results)==0 :
        return [] ,[] ,[],[],[]
    face_array=[]
    x1_frame=[]
    y1_frame=[]
    x2_frame=[]
    y2_frame=[]
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))
        x1_frame.append(x1)
        y1_frame.append(y1)
        x2_frame.append(x2)
        y2_frame.append(y2)
    return face_array,x1_frame,y1_frame,x2_frame,y2_frame





def try_performTest(model,trainX,face,clf):
  #face,x1,y1,x2,y2 = extract_face_webcam(path)
  testX = get_embedding(model, face)
  testX = testX.reshape((1,128))
  concat = np.concatenate((trainX,testX),axis=0)
  lent = len(trainX)
  in_encoder = Normalizer(norm='l2')
  concat = in_encoder.transform(concat)
  trainX = concat[:lent]
  testX = concat[lent:]
  
  predicted = clf.predict(testX)
  
  if(np.max(clf.predict_proba(testX))>0.2):
    return predicted
  else:
    return list()