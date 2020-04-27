import sys
from termcolor import colored,cprint

import stdiomask
import functions
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os.path import isdir
from os import listdir
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import pickle
import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
from db.CourseLoginLayer import CourseAuthenticator
from db.AttendanceLayer import Attendance
from db.StudentLayer import Student
from db.course_reg import Course_Reg
from db.SleepLogger import DrowsyDetector
import datetime
from scipy.spatial import distance
from imutils import face_utils
import dlib



class detectDrowsiness:
	def __init__(self,classRoom):
		self.Model               =   None
		self.kerasPath           =   "./model/facenet_keras.h5"
		self.trainPath           =   "./model/train.pickle"
		self.testPath            =   "./model/test.pickle"
		self.trainLabelPath      =   "./model/train_labels.pickle"
		self.testLabelPath       =   "./model/test_labels.pickle"
		self.classifierPath      =    "./model/classifier.pickle"
		self.trainingData         =    None
		self.testingData          =    None
		self.train_labels        =    None
		self.trainingData         =    None
		self.test_labels         =    None
		self.classifier          =    None
		self.studentIDs          =    set()
		self.class_room          =    classRoom
		self.sleeping_students   =    np.array([])


	def verifyCourseIDandPassKey(self,courseID,PassKey):
		return CourseAuthenticator().verifyAuth(courseID,PassKey)


	def loadStudents(self,courseID):
		now = datetime.datetime.now()
		Course_dict = {}
		Course_dict['Semester_year'] = now.year
		if (now.month<=6):
			Course_dict['Semester_type'] = 'Spring'
		else:
			Course_dict['Semester_type'] = 'Monsoon'
		Course_dict['course_id'] = courseID
		self.studentIDs = set(Course_Reg().get_rollnumber(Course_dict))
		if(len(self.studentIDs)>0):
			return True
		else:
			return False


	def eye_aspect_ratio(self, eye):
		A = distance.euclidean(eye[1], eye[5])
		B = distance.euclidean(eye[2], eye[4])
		C = distance.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)
		return ear

	def checkDrowsiness(self,courseID):
		thresh = 0.25
		frame_check = 30
		detect = dlib.get_frontal_face_detector()
		predict = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
		cap=cv2.VideoCapture(0)
		flag=0
		while True:
			ret, frame=cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			subjects = detect(gray, 0)
			for subject in subjects:
				shape = predict(gray, subject)
				shape = face_utils.shape_to_np(shape) #converting to NumPy Array
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = self.eye_aspect_ratio(leftEye)
				rightEAR = self.eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				#cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				#cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				if ear < thresh:
					flag += 1
					#print (flag)
					if flag >= frame_check:
						value = frame
						cv2.putText(value, "****************ALERT!****************", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.putText(value, "****************ALERT!****************", (10,325),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						self.retrieveRollNumber(frame, courseID)
				else:
					flag = 0
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		cprint("CLASS HAS ENDED", 'green', attrs=['reverse', 'bold'])
		DrowsyDetector().markStudentSleepy(list(map(int, self.sleeping_students.tolist())), courseID)
		cv2.destroyAllWindows()
		cap.stop()
		#print(self.sleeping_students)

	def retrieveRollNumber(self,frame,courseID):
		frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
		face_array,x1_frame,y1_frame,x2_frame,y2_frame = functions.try_extract_face_webcam(frame)

		if len(face_array) == 0:
			print("No face Detected")
		else:
			for face in range(len(face_array)):
				predict = functions.try_performTest(self.model,self.trainingData,face_array[face],self.classifier)

				#print(predict)
				if(len(predict)) == 0:
					print(" face Not Recognized")
					#pass
				else:
					#print("HELLO?", predict)
					self.sleeping_students = np.append(self.sleeping_students,predict)
					#print("HELLO?", self.sleeping_students)
					self.sleeping_students = np.unique(self.sleeping_students)



	def loadModule(self):
		self.model = load_model(self.kerasPath,compile=False)
		print("model loaded")

		with open(self.trainPath, 'rb') as f:
			self.trainingData = pickle.load(f)
		print("trainingData loaded")

		with open(self.testPath, 'rb') as f:
			self.testingData = pickle.load(f)
		print("testingData loaded")

		with open(self.trainLabelPath, 'rb') as f:
			self.train_labels = pickle.load(f)
		print("train_labels loaded")

		with open(self.testLabelPath, 'rb') as f:
			self.test_labels = pickle.load(f)
		print("test_labels loaded")

		with open(self.classifierPath, 'rb') as f:
			self.classifier = pickle.load(f)
		print("classifier loaded")

		return True


cprint("Drowsiness Detection System Initiated", 'green', attrs=['reverse', 'bold'])
courseID   =    input("Enter Course ID : ")
PassKey      = stdiomask.getpass(prompt='Enter pass key for user :', mask='*')
#PassKey       =    input("Enter pass key for user : ")
detectSleepObj  =    detectDrowsiness(input("Enter Class room : "))

if (detectSleepObj.verifyCourseIDandPassKey(courseID,PassKey)):

	if (detectSleepObj.loadStudents(courseID)):

		if(detectSleepObj.loadModule()):

			detectSleepObj.checkDrowsiness(courseID)

		else:
			cprint("Some Internal Error occurs with the module,please re-start again" ,'red', attrs=['reverse', 'bold'])
			exit()

	else:
		cprint("Problem In Loading Student,Contact DataBase Administrator", 'red', attrs=['reverse', 'blink'])
		exit()

else:
	cprint("Input Values is not valid,Start the Module again", 'red', attrs=['reverse', 'blink'])
	exit()
