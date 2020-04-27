import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
from datetime import datetime
from db.SleepLogger import DrowsyDetector

class Attendance:
    

	def __init__(self,info_dict={}):
		if(len(info_dict.keys())!=0):
			self.course_id=info_dict['course_id']
			self.roll_number=info_dict['roll_number']
			self.full_name=info_dict['full_name']
			#self.course_name=info_dict['course_name']
			self.semester_type=info_dict['semester_type']
			self.semester_year=info_dict['semester_year']
			self.class_room=info_dict['class_room']


	def markAttendance(self):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			insert_query='''insert into attendance(course_id,roll_number,
			full_name,semester_year,semester_type,date_of_class,time_of_class,
			class_room) values (%s,%s,%s,%s,%s,%s,%s,%s)'''
			cursor = conn.cursor()
			##mysql date format is YYYYMMDD
			##mysql time format is HH:MM:SS
			now = datetime.now()
			today_date = now.strftime("%Y/%m/%d")
			today_time=now.strftime("%H:%M:%S")
			record_tuple=(self.course_id,self.roll_number,self.full_name,self.semester_year,self.semester_type,today_date,today_time,self.class_room)
			cursor.execute(insert_query, record_tuple)
			conn.commit()
			print('Student ',self.full_name,' Attendance Marked Successfully!')
			#conn.close()
			#cursor.close()

		except mysql.connector.Error as error:
			print("Failed to insert into MySQL table {}".format(error))

		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()


	def getAttendanceByDate(self,info_dict):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			sql_select_query='''select count(*) from attendance where 
			course_id = %s and roll_number= %s and semester_year= %s and semester_type=%s
			and date_of_class=%s '''
			cursor=conn.cursor()
			record_tuple=(info_dict['course_id'],info_dict['roll_number'],info_dict['semester_year'],info_dict['semester_type'],info_dict['date_of_class'])
			cursor.execute(sql_select_query,record_tuple)
			records = cursor.fetchall()
			ret_val=''
			for row in records:
				if(row[0]>=1):
					ret_val='True'
				break
		except Error as e:
			print("Error reading data from MySQL table", e)

		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()
			return ret_val

	def getAttendanceByMonth(self,info_dict):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			sql_select_query='''select count(*) from attendance where 
			course_id = %s and roll_number= %s and semester_year= %s and semester_type=%s
			and MONTH(date_of_class)=%s '''
			cursor=conn.cursor()
			record_tuple=(info_dict['course_id'],info_dict['roll_number'],info_dict['semester_year'],info_dict['semester_type'],info_dict['month'])
			cursor.execute(sql_select_query,record_tuple)
			records=cursor.fetchone()
			ats_count=0
			if(len(records)>=1):
				ats_count=records[0]
		except Error as e:
			print("Error reading data from MySQL table",e)
		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()
			return ats_count

	def getGrossAttendance(self,info_dict):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			sql_select_query='''select count(*) from attendance where 
			course_id = %s and roll_number= %s and semester_year= %s and semester_type=%s
			and YEAR(date_of_class)=%s '''
			cursor=conn.cursor()
			record_tuple=(info_dict['course_id'],info_dict['roll_number'],info_dict['semester_year'],info_dict['semester_type'],info_dict['year'])
			cursor.execute(sql_select_query,record_tuple)
			records=cursor.fetchone()
			ats_count=0
			if(len(records)>=1):
				ats_count=records[0]
		except Error as e:
			print("Error reading data from MySQL table",e)
		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()
			return ats_count

	def getNetAttendance(self,info_dict):
		drowsyObject=DrowsyDetector()
		return (self.getGrossAttendance(info_dict)-drowsyObject.calculatePenalty(info_dict['roll_number'],info_dict['course_id']))
 
'''USAGE Example'''

# inf_dict={}
# inf_dict['course_id']='CSE427'
# inf_dict['course_name']='POIS'
# inf_dict['roll_number']=5
# inf_dict['full_name']='Tushar'
# inf_dict['semester_year']=2019
# inf_dict['semester_type']='Monsoon'
# inf_dict['class_room']='H105'
# inf_dict['date_of_class']='2020-04-15'
# inf_dict['year']=2020

# ats=Attendance(inf_dict)
# print(ats.getGrossAttendance(inf_dict))
# print(ats.getNetAttendance(inf_dict))

'''This class has constructor that takes dictionary with
the following keys
key    				Value
course_id			String value of courseid
roll_number			integer roll number
full_name			full name of student as string
semester_type		type of semester as string
semester_year		integer value of semester year
class_room			Room of class as string value(like 'H205')

(1)markAttendance is used to mark the attendance of the student whose fields are initialised
in the keys of dictionary passed to constructor.
Usage:Shown below
(2)getAttendanceByDate also take input in form of dictionary with following key fields
key    				Value
course_id			String value of courseid
roll_number			integer roll number
semester_type		type of semester as string
semester_year		integer value of semester year
class_room			Room of class as string value(like 'H205')
date_of_class		Value being of form 'YYYY-MM-DD'

This functions returns True if the student was present on a particular date in a particular
subject class.
(3)getAttendanceByMonth expects dictionary with key fields of form:
key    				Value
course_id			String value of courseid
roll_number			integer roll number
semester_type		type of semester as string
semester_year		integer value of semester year
month				Value being of form type int denoting 
					for which month we want to check attendance.(eg April-4)

This function returns the total number of classes attended by a particular student
in particular course in a particular month.

(4)getGrossAttendance()-This function expects a dictionary with key and values as:
key    				Value
course_id			String value of courseid
roll_number			integer roll number
semester_type		type of semester as string
semester_year		integer value of semester year
year				Value being of form type int denoting 
					for which year we want to check attendance.(eg 2019)
Then this function returns a vlue indicating how many classes till date
student has attended for the subject for which course id was given.
Note that this does not count the penalty.


(5)getNetAttendance()-Expects parameters in same form as getGrossAttendance(),
but this returns total attendance of student after deductions based on his
Drowsyness score.'''

'''

#Usage type example below


# inf_dict={}
# inf_dict['course_id']='CSE427'
# inf_dict['course_name']='POIS'
# inf_dict['roll_number']=5
# inf_dict['full_name']='Tushar'
# inf_dict['semester_year']=2019
# inf_dict['semester_type']='Monsoon'
# inf_dict['class_room']='H105'
# inf_dict['date_of_class']='2020-04-15'
# inf_dict['month']=4

# ats=Attendance(inf_dict)
# ats.markAttendance()
# print(ats.getAttendanceByDate(inf_dict))
# print(ats.getAttendanceByMonth(inf_dict))'''



			
