import os
import sys
import pickle
import yaml
from src.exception import CustomException
from src.logger import logging

def save_object(file_path:str,obj):
    '''
        Desc: This function responsible for saving our model and preprocessor in pickle
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def read_yaml_file(file_path:str)->dict:
    '''
        Desc: This function responsible for reading our yaml file
    '''
    try:
        with open(file_path,'rb') as yaml_obj:
            return yaml.safe_load(yaml_obj)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    '''
        Desc: This function will read or load our pickled file
    '''
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj) 
    except Exception as e:
        raise CustomException(e,sys)