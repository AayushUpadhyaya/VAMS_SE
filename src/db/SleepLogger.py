import mysql.connector
from mysql.connector import Error
from datetime import datetime
from db.DBService import MyDatabase
PERIOD_LENGTH=5
class DrowsyDetector:
    
    def markStudentSleepy(self,roll_list,course_id):
        date=datetime.now().strftime("%Y/%m/%d")
        if(len(roll_list)!=0):
            try:
                mydb = MyDatabase()
                conn = mydb.get_connection()
                cursor = conn.cursor()
                for roll_numb in roll_list:
                    cursor.execute("insert into drowsy_data(course_id,roll_number,date_of_class) values(%s,%s,%s);",(course_id,roll_numb,date))
                conn.commit()
                #print('Successfully recorded the details of students found sleepy')

            except mysql.connector.Error as error:
              print("Failed to insert into MySQL table {}".format(error))

            finally:
              if(conn.is_connected()):
                conn.close()
                cursor.close()    
    
    def getSleepCount(self,roll_number,course_id):
        try:
            mydb = MyDatabase()
            conn = mydb.get_connection()
            cursor = conn.cursor()
            select_query="select count(*) from drowsy_data where course_id=(%s) and roll_number=(%s)"
            record_tuple=(course_id,roll_number)
            cursor.execute(select_query,record_tuple)
            record=cursor.fetchone()
            
        except Error as e:
            print("Error reading data from MySQL table", e)

        finally:
            if(conn.is_connected()):
                conn.close()
                cursor.close()
            return record[0]
        

    def calculatePenalty(self,roll_number,course_id):
        total_sleep=self.getSleepCount(roll_number,course_id)
        return (total_sleep)//PERIOD_LENGTH
    
    
'''USAGE INFORMATION
This particular file can be used for marking students who found sleepy in the class.
markStudentSleepy():
This method expects a set of rollnumbers found sleepy in the class(will be used by the
drowsy_detection module at the end of class,by noting all the roll numbers in a python set,so
that there will be no issues of redundant data) along with the course-Id and inserts the sent
information into the drowsy_data for further actions. 
'''

'''getSleepCount()-This function expects a roll number and course_id as parameter and
returns a int value indicating in how many classes till date, student found
sleeping.

calculatePenalty()-This function expects a roll number and course_id as parameter and
returns a value that must be deducted from student's total attendance as penalty'''