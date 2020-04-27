import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase

class Course_Reg:
    
    def __init__(self,info_dict={}):
        if(len(info_dict.keys())!=0):
            self.course_id=info_dict['course_id']
            self.roll_number=info_dict['roll_number']
            self.Semester_type=info_dict['Semester_type']
            self.Semester_year=info_dict['Semester_year']
            
    def register_course(self):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("insert into course_reg(course_id,roll_number,Semester_type,Semester_year) values(%s,%s,%s,%s);",(self.course_id,self.roll_number,self.Semester_type,self.Semester_year))
            conn.commit()
            print('student with roll_number',self.roll_number,'registered for Course ',self.course_id,'Successfully!')

        except mysql.connector.Error as error:
          print("Failed to insert into MySQL table {}".format(error))

        finally:
          if(conn.is_connected()):
            conn.close()
            cursor.close()

    def get_rollnumber(self,query_dict):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("select roll_number from course_reg where course_id=(%s) and Semester_type=(%s) and Semester_year=(%s);",(query_dict['course_id'],query_dict['Semester_type'],query_dict['Semester_year']))
            res=cursor.fetchall()
            roll_numb=[]
            for numb_tup in res:
                roll_numb.append(numb_tup[0])
                
        except Error as e:
            print("Error reading data from MySQL table", e)

        finally:
            if(conn.is_connected()):
                conn.close()
                cursor.close()      
            return roll_numb

    def get_courseid(self,query_dict):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            cursor.execute("select course_id from course_reg where roll_number=(%s) and Semester_type=(%s) and Semester_year=(%s);",(query_dict['roll_number'],query_dict['Semester_type'],query_dict['Semester_year']))
            res=cursor.fetchall()
            courses=[]
            for course in res:
                courses.append(course[0])
                
        except Error as e:
            print("Error reading data from MySQL table", e)

        finally:
            if(conn.is_connected()):
                conn.close()
                cursor.close()         
            return courses

'''USAGE INFORMATION
The constructor of this class expects dictionary with keys and values as
Key                 Value
course_id           Unique Id of the Course
roll_number         Roll Number assigned to the Student
Semester_type       Should Specify whether its monsoon or spring Semester
Semester_year       The current Semester year of the Student.
(1)register_course()-This method adds the courses entries registered by students into the
course_reg table taking all the above mentioned fields as input.
(2)get_rollnumber()-This method expects course_id,Semester_type and year as arguments and
it returns the students roll numbers registered for that course Id.
(3)get_course_id()-This method expects roll number,Semester_type and year as arguments and
it returns the courses the student with roll number passed has taken.
'''

#USAGE Example

#infodict={'course_id':'cse123','roll_number':12,'Semester_type':'monsson','Semester_year':1}
#ss=Course_Reg(infodict)
#ss.register_course()
#querydict={'course_id':'cse123','Semester_type':'monsson','Semester_year':1}
#print(ss.get_rollnumber(querydict))
#querydict1={'roll_number':12,'Semester_type':'monsson','Semester_year':1}
#print(ss.get_course_id(querydict1))
 
        

        
        
        
        
    
