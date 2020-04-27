import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
class Courses:

    def __init__(self,info_dict={}):
        if(len(info_dict.keys())!=0):
            self.course_id=info_dict['course_id']
            self.course_name=info_dict['course_name']

    def add_course(self):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("insert into courses(course_id,course_name) values(%s,%s);",(self.course_id,self.course_name))
            conn.commit()
            print('Course ',self.course_id,' Added Successfully!')

        except mysql.connector.Error as error:
          print("Failed to insert into MySQL table {}".format(error))

        finally:
          if(conn.is_connected()):
            conn.close()
            cursor.close()


    def get_course_name(self,course_id):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("select course_name from courses where course_id=(%s);",(course_id,))
            record=cursor.fetchone()
            
        except Error as e:
            print("Error reading data from MySQL table", e)

        finally:
            if(conn.is_connected()):
                conn.close()
                cursor.close()
            return record[0]
        

    def get_course_id(self,course_name):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("select course_id from courses where course_name=(%s)",(course_name,))
            record=cursor.fetchone()
        except Error as e:
            print("Error reading data from MySQL table", e)

        finally:
            if(conn.is_connected()):
                conn.close()
                cursor.close()
            return record[0]
    
'''USAGE INFORMATION
The constructor of this class expects dictionary with keys and values as
Key                 Value
course_id           Unique Id of the Course
course_name         Name of the Course corresponding to the Course-ID

(1)add_course()-This method adds the course entry into the Courses table,which maintains the
tuples with attributes CourseId and its Name.
(2)get_course_name()-This method expects course Id as an argument and it returns the
corresponding course name assigned to that course Id.
(3)get_course_id()-This method expects course name as an argument and it returns the
corresponding course Id assigned to that course name.
'''

#USAGE Example

#infodict={'course_id':'cse123','course_name':'SMAI'}
#ss=Courses(infodict)
#ss.add_course()
#print(ss.get_course_id('Parallel'))
#print(ss.get_course_name('cse234'))
   
        
