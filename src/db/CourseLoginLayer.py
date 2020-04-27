import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
class CourseAuthenticator:

	def insertCourseAuthDetails(self,course_id,passkey):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			insert_query='''insert into course_credentials values (%s,SHA1(%s));'''
			cursor = conn.cursor()
			record_tuple=(course_id,passkey)
			cursor.execute(insert_query, record_tuple)
			conn.commit()
			print('Course  ',course_id,' passkey created successfully!')
			conn.close()
			cursor.close()
		except mysql.connector.Error as error:
			print("Failed to insert into MySQL table {}".format(error))
		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()

	def verifyAuth(self,course_id,passkey):
		try:
			mydb=MyDatabase()
			conn=mydb.get_connection()
			sql_select_query='''select count(*) from course_credentials where course_id = %s 
			and passkey=SHA1(%s);'''
			cursor=conn.cursor()
			record_tuple=(course_id,passkey)
			cursor.execute(sql_select_query,record_tuple)
			records = cursor.fetchone()
			ret_val=0
			if(records[0]>=1):
				ret_val=1
		except Error as e:
			print("Error reading data from MySQL table", e)
		finally:
			if(conn.is_connected()):
				conn.close()
				cursor.close()
			return ret_val
'''USAGE DETAILS



obj=CourseAuthenticator() --> create object


obj.insertCourseAuthDetails('CSE427','ayush123')--> This is how you
enter course credential for registration.

--->How verification is done is below here.
print(obj.verifyAuth('CSE427','ayush')) #returns 0
print(obj.verifyAuth('CSE427','ayush123')) #returns 1
'''