"""
step 1: enter course ID
step 2: load ta roll numbers and instructor of that course
step 3: enter ta roll number
step 4: match the TA roll number with the loaded rollnumber
step 5: if (match) else close system send security message
step 6 :load all student roll number
step 7 : loop
step 8 :take picture
step 9 :predict picture
step 10: if (match) else ignore
step 11 : mark attendance of that student

"""
import sys
from termcolor import colored,cprint
import stdiomask
import facialExtractionHelpers
import numpy as np
import cv2
from os.path import isdir
from os import listdir
from numpy import load
from keras.models import load_model
import pickle
import mysql.connector
from mysql.connector import Error
from databaseHandlers.DBService import MyDatabase
from databaseHandlers.CourseLoginLayer import CourseAuthenticator
from databaseHandlers.AttendanceLayer import Attendance
from databaseHandlers.StudentLayer import Student
from databaseHandlers.course_reg import Course_Reg
from databaseHandlers.SleepLogger import DrowsyDetector
import datetime


class virtualAttendance:
    def __init__(self,classRoom):
        self.Model               =   None
        self.kerasPath           =   "../model/facenet_keras.h5"
        self.trainPath           =   "../model/train_data.pickle"
        self.testPath            =   "../model/test_data.pickle"
        self.trainLabelPath      =   "../model/train_labels.pickle"
        self.testLabelPath       =   "../model/test_labels.pickle"
        self.classifierPath      =    "../model/classifier.pickle"
        self.testingData          =    None
        self.train_labels        =    None
        self.trainingData         =    None
        self.test_labels         =    None
        self.classifier          =    None
        self.studentID=set()
        self.class_room = classRoom


    def verifyCourseIDandPassKey(self,courseID,PassKey):
        return CourseAuthenticator().verifyAuth(courseID,PassKey)

    def loadStudent(self,courseID):
        now = datetime.datetime.now()
        Course_dict = {}
        Course_dict['Semester_year'] = now.year
        if (now.month<=6):
            Course_dict['Semester_type'] = 'Spring'
        else:
            Course_dict['Semester_type'] = 'Monsoon'
        Course_dict['course_id'] = courseID
        self.studentID = set(Course_Reg().get_rollnumber(Course_dict))
        if(len(self.studentID)>0):
            return True
        else:
            return False

    def markAttendance(self,predicts,courseID):
        for predict in predicts:
            predict = int(predict)
            if(predict in self.studentID):
                info_dict = {}
                info_dict['course_id'] = courseID
                info_dict['roll_number'] = predict
                info_dict['full_name'] = Student().getFullName(predict)
                now = datetime.datetime.now()
                info_dict['semester_year'] = now.year
                if (now.month<=6):
                    info_dict['semester_type'] = 'Spring'
                else:
                    info_dict['semester_type'] = 'Monsoon'
                info_dict['class_room'] = self.class_room
                Attendance_Obj = Attendance(info_dict)
                Attendance_Obj.markAttendance()

            else:
                print("Student not recognised")


    def startAttendance(self,courseID):
        cap = cv2.VideoCapture(0)
        while(True):
            ret,frame = cap.read()
            frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
            face_array,x1_frame,y1_frame,x2_frame,y2_frame = facialExtractionHelpers.try_extract_face_webcam(frame)

            if len(face_array) == 0:
                print("No face Detected")
                continue
            else:
                for face in range(len(face_array)):
                    predict = facialExtractionHelpers.try_performTest(self.model,self.trainingData,face_array[face],self.classifier)

                    if(len(predict)) == 0:
                        print(" Face Not Recognized")
                        continue

                    self.markAttendance(predict,courseID)

                    color =  (0 , 0, 255)
                    stroke = 5
                    cv2.rectangle(frame , (x1_frame[face],y1_frame[face]) ,(x2_frame[face],y2_frame[face]),color,stroke)

                cv2.imshow('frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    def loadModule(self):
        self.model = load_model(self.kerasPath,compile=False)
        print("Model loaded")

        with open(self.trainPath, 'rb') as f:
            self.trainingData = pickle.load(f)
        print("Training Data loaded")

        with open(self.testPath, 'rb') as f:
            self.testingData = pickle.load(f)
        print("Testing Data loaded")

        with open(self.trainLabelPath, 'rb') as f:
            self.train_labels = pickle.load(f)
        print("Training labels loaded")

        with open(self.testLabelPath, 'rb') as f:
            self.test_labels = pickle.load(f)
        print("Testing labels loaded")

        with open(self.classifierPath, 'rb') as f:
            self.classifier = pickle.load(f)
        print("Classifier loaded")

        return True


def main():
    cprint("Attendance System Initiated", 'green', attrs=['reverse', 'bold'])
    courseID   =    input("Enter Course ID : ")
    PassKey      = stdiomask.getpass(prompt='Enter pass key for user :', mask='*')

    virtualAtt  =    virtualAttendance(input("Enter Class room : "))

    if (virtualAtt.verifyCourseIDandPassKey(courseID,PassKey)):

        if (virtualAtt.loadStudent(courseID)):

            if(virtualAtt.loadModule()):

                virtualAtt.startAttendance(courseID)


            else:
                cprint("Some Internal Error occurs with the module,please re-start again" ,'red', attrs=['reverse', 'bold'])
                exit()

        else:
            cprint("Problem In Loading Student,Contact DataBase Administrator", 'red', attrs=['reverse', 'blink'])
            exit()

    else:
        cprint("Input Values is not valid,Start the Module again", 'red', attrs=['reverse', 'blink'])
        exit()

if __name__ == "__main__":
    main()
