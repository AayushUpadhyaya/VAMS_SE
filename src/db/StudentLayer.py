import mysql.connector
from mysql.connector import Error
from db.DBService import MyDatabase
class Student:

    def __init__(self, input_dict={}):
    	if(len(input_dict.keys())!=0):  
        	self.name = input_dict['name']
        	self.facial_features=b'0'

#facial_features are optional.

    def createStudent(self):
        try:
          mydb=MyDatabase()
          conn=mydb.get_connection()
          insert_query='''insert into students(full_name,facial_features) values (%s,%s)'''
          cursor = conn.cursor()
          record_tuple=(self.name,self.facial_features)
          next_num=self.getNextRollNumber()
          cursor.execute(insert_query, record_tuple)
          conn.commit()
          print('Student ',self.name,' Registered Successfully!')
          print(self.name,' has roll number : ',next_num)
          conn.close()
          cursor.close()

        except mysql.connector.Error as error:
          print("Failed to insert into MySQL table {}".format(error))

        finally:
          if(conn.is_connected()):
            conn.close()
            cursor.close()


    def getFullName(self,roll_number):
      try:
        mydb=MyDatabase()
        conn=mydb.get_connection()
        sql_select_query='''select full_name from students where roll_number = %s'''
        cursor=conn.cursor()
        
        cursor.execute(sql_select_query,(roll_number,))
        records = cursor.fetchall()
        for row in records:
          #print (row)
          ret_val=row[0]
          break
      except Error as e:
        print("Error reading data from MySQL table", e)

      finally:
        if(conn.is_connected()):
          conn.close()
          cursor.close()
        return ret_val


    def getFacialFeatures(self,roll_number):
      try:
        mydb=MyDatabase()
        conn=mydb.get_connection()
        sql_select_query='''select facial_features from students where roll_number = %s'''
        cursor=conn.cursor()
        
        cursor.execute(sql_select_query,(roll_number,))
        records = cursor.fetchall()
        for row in records:
          #print (row)
          ret_val=row[0]
          break
      except Error as e:
        print("Error reading data from MySQL table", e)

      finally:
        if(conn.is_connected()):
          conn.close()
          cursor.close()
        return ret_val

    def getAllStudentRecord(self):
    	try:
    		roll_number=[]
    		name=[]
    		mydb=MyDatabase()
    		conn=mydb.get_connection()
    		sql_select_query='''select roll_number,full_name from students;'''
    		cursor=conn.cursor()
    		cursor.execute(sql_select_query)
    		records = cursor.fetchall()
    		for row in records:
    			roll_number.append(row[0])
    			name.append(row[1])
    	except Error as e:
    		print("Error reading data from MySQL table", e)
    	finally:
    		if(conn.is_connected()):
    			conn.close()
    			cursor.close()
    			return roll_number,name

    def getNextRollNumber(self):
    	try:
    		mydb=MyDatabase()
    		conn=mydb.get_connection()
    		sql_select_query='''SELECT AUTO_INCREMENT FROM information_schema.TABLES
    		 WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s;'''
    		record_tuple=('VAMS','students')
    		cursor=conn.cursor()
    		cursor.execute(sql_select_query,record_tuple)
    		records = cursor.fetchone()
    		ret_val=records[0]
    	except Error as e:
    		print("Error reading data from MySQL table", e)
    	finally:
    		if(conn.is_connected()):
    			conn.close()
    			cursor.close()
    		return ret_val




'''USAGE INFORMATION
The constructor of this class expects dictionary with keys and values as
Key                 Value
name                Full name of student
facial_features(optional)     facial embedding vector(1X128) of face of student
                    passed as sequence of bytes.

(1)createStudent()-This method creates student with details contained in the dictionary 
passed to the constructor.Roll Number and Date of Creation are auto-assigned.
(2)getFullName(roll_number)-This method expects an integer argument that corresponds
to the roll number of student and returns the full name corresponding to that roll_number.
(3)getFacialFeatures(roll_number)-This method expects an integer argument that corresponds
to the roll number of the student and returns the facial feature embedding vector as sequence
of bytes.You need to convert it to numpy array for further processing.
(4)getNextRollNumber()-Gives an integer output indicating what next roll number
will be assigned to a new student.This function does not expect an argument.
'''

'''NOTE: Constructor has to be passed dictionary with key only 'name'
corresponding to full name of the student when you want to create a student otherwise
nothing needs to be passed to the constructor'''


#USAGE Example

#mydict={'name':'Ramesh','facial_features':b'89'}
#ss=Student(mydict)
#ss.createStudent()
#r,fn=ss.getAllStudentRecord()
#print(r)
#print(fn)
#print(ss.getNextRollNumber())




			
			