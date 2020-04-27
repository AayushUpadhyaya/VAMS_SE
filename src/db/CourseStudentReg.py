import db.course_reg as cr
from datetime import datetime

roll_number=input("Enter your roll number:")
sem_yr=input("Enter your Semester year:")
sem_mnth=datetime.now().month

if(sem_mnth<=7 and sem_mnth>=1):
    sem_type='Spring'
else:
    sem_type='Monsoon'

no_courses=input("enter the number of courses that you want to register for this semester")

info_dict={}
info_dict['roll_number']= roll_number
info_dict['Semester_type']=sem_type
info_dict['Semester_year']=sem_yr

for index in range(int(no_courses)):
    cid=(input("enter the course id:"))
    info_dict['course_id']=cid
    reg_obj=cr.Course_Reg(info_dict)
    reg_obj.register_course()

'''USAGE INFORMATION
This particualr file can be used for making students register for the courses they
wish at the beginning of their semesters.The file expects the roll number of the
student and the courses he/she wishes to take in that semester and helps them in
registering the course!!
'''
