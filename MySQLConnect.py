"""
Central module for handling MySQL connection.
This module uses SQLAlchemy to establish a connection to a MySQL server. An env file is used to store server information such as hostname, username, password, and database, and an engine is returned which can store, retrieve, and manipulate data in that database.

"""
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


def mysql_engine():
    #Function to establish a connection to a MySQL server
    
    #Extracting variable names from `.env` file
    
    load_dotenv("./MySQL-config.env")
    mysql_host = os.environ.get("MYSQL_HOST")
    mysql_user = os.environ.get("MYSQL_USER")
    mysql_password = os.environ.get("MYSQL_PASSWORD")
    mysql_database = os.environ.get("MYSQL_DATABASE")
    
    
    #Establishing connection to database using the variable names
    connection_string = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}"
    engine = create_engine(connection_string)
    
    return engine