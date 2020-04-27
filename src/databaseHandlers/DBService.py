import mysql.connector
from mysql.connector import Error
class MyDatabase:

	def get_connection(self):
		try:
			connection = mysql.connector.connect(host='localhost',database='VAMS',user='ayush',password='1234')
		except Error as e:
			print("Error connecting to database")
		finally:
			if (connection.is_connected()):
				return connection
'''This class's method get_connection provides a connection object to mysql database'''