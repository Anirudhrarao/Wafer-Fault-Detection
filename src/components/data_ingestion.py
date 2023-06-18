import os 
import sys
import pandas as pd 
import numpy as np
from pymongo import MongoClient
from zipfile import Path
from src.constant import DATABASE_NAME, COLLECTION_NAME, MONGO_DB_URL
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
# from src.utils import MainUtils
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    artifact_folder = os.path.join('artifact')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        # self.utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name, db_name):
        '''
            Desc: This function will export data which stored in our mongodb as DataFrame
        '''
        try:
            # creating instance for MongoClient class and making connection with db
            logging.info('Entered into data ingestion')
            mongo_client = MongoClient(MONGO_DB_URL)
            logging.info('Connecting Mongodb Atlas...')
            collection = mongo_client[db_name][collection_name]
            # reading collection as data frame
            df = pd.DataFrame(list(collection.find()))
            logging.info('Collection read as data frame')
            # removing "_id" columns from dataset
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            # replacing na which is use in mongodb as none but in python we use np.nan
            df.replace({"na":np.nan},inplace=True)
            logging.info('Exit from data ingestion')
            return df

        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_file_path(self)->pd.DataFrame:
        '''
            Desc: This method reads data from mongodb and saves it into artifacts.
        '''
        try:
            logging.info('Exporting data from mongodb')
            # Taking folder path we initialized in data ingestion before
            raw_file_path = self.data_ingestion_config.artifact_folder
            # creating artifact folder
            os.makedirs(raw_file_path,exist_ok=True)
            sensor_data = self.export_collection_as_dataframe(
                db_name=DATABASE_NAME,
                collection_name=COLLECTION_NAME
            )
            logging.info(f'Saving exported data into artifact folder {raw_file_path}')

            store_file_path = os.path.join(raw_file_path,'wafer_fault.csv')
            sensor_data.to_csv(store_file_path,index=False)
            return store_file_path
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self)->Path:
        '''
            Desc: This method initiates the data ingestion components of training pipeline
        '''
        logging.info('Entered into initiate data ingestion method of data ingestion class')
        try:
            store_file_path = self.export_data_into_file_path()
            logging.info('Got the data from mongodb under artifact folder')
            logging.info('Exited initiate data ingestion method of data ingestion class')

            return store_file_path
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    store_file_path = obj.initiate_data_ingestion()
    dt = DataTransformation(store_file_path=store_file_path)
    dt.initiate_data_transformation()
