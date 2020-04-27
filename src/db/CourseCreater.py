'''This python code presents a menu based interface to
add courses, get information about a particular offered course.'''


import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
from cprint import *
from db.courses import Courses
import sys

cprint.info('Welcome to Course Catalog Manager')
cprint('Press the key corresponding to option number to continue.')
cprint('1: Add a course')
cprint('2: Get course Id of a course')
cprint('3:Get Course Name of a course')
cprint('Any other key: exit')
user_choice=input('')
if int(user_choice)==1:
    course_id=input('Enter course_id\n')
    course_name=input('Enter course name\n')
    info_dict={'course_id':course_id,'course_name':course_name}
    cObj=Courses(info_dict)
    cObj.add_course()
    cprint.info('Course added successfully!')
elif int(user_choice)==2:
    course_name=input('Enter course name\n')
    cObj=Courses()
    print('Course Id is: ',cObj.get_course_id(course_name))
    
elif int(user_choice)==3:
    course_id=input('Enter course id\n')
    cObj=Courses()
    print('Course Name is : ',cObj.get_course_name(course_id))
    
else:
    cprint('Bye')
    sys.exit()
    
