import os
from pathlib import Path

def get_current_directory():
    current_directory = os.getcwd()
    current_directory = Path(current_directory)
    return current_directory

def create_directory(create_path):
    try:
        os.mkdir(create_path)
    except OSError:  
        print ("Creation of the directory %s failed" % create_path)
    else:  
        print ("Successfully created the directory %s " % create_path)
        
