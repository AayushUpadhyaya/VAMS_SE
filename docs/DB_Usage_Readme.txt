(1)Install mysql on system
(2)Install python mysql connector as
pip3 install mysql-connector-python
(3)Open terminal and enter mysql prompt by 'sudo mysql'
(4)create user ayush as
CREATE USER 'ayush'@'localhost' IDENTIFIED BY '1234';
(5)Grant all permissions
GRANT ALL PRIVILEGES ON * . * TO 'ayush'@'localhost';
(6)type 'exit' to exit mysql prompt.
(7)After returning back to shell, type below
mysql -u ayush -p
(8)create the database VAMS as
create database VAMS;
(9)start using the database by typing below command
use VAMS;
(10)create all the tables by executing the commands in 'database_schema.txt'
(11)All set!!
