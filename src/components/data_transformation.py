import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.constant import DATABASE_NAME,COLLECTION_NAME,MONGO_DB_URL
from src.utils.utils import save_object

import pandas as pd 
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    artifact_folder = os.path.join('artifact')
    train_file_path = os.path.join(artifact_folder,'train.npy')
    test_file_path = os.path.join(artifact_folder,'test.npy')
    preprocessor_object_file = os.path.join(artifact_folder,'preprocessor.pkl')

class DataTransformation:
    def __init__(self,store_file_path):
        self.data_transformation_config = DataTransformationConfig()
        self.store_file_path = store_file_path

    
    @staticmethod
    def get_data(store_file_path:str) -> pd.DataFrame:
        '''
            Desc: This method will read raw data from artifact folder
        '''
        try:
            # Reading dataset from path
            data = pd.read_csv(store_file_path)
            # renaming columns "Good/Bad" with Quality
            data.rename(columns={"Good/Bad": "Quality"},inplace=True)
            return data 
        
        except Exception as e:
            raise CustomException(e,sys)


    def get_data_transformation_obj(self):
        '''
            Desc: This function will transform our data fill missing value and scaling data
        '''
        try:
            preprocessor = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='constant',fill_value=0)),
                    ('scaler',RobustScaler())
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        '''
            Desc: This method initiates the data transformation component for the pipeline
        '''
        logging.info('Entered into initiate data transformation method of data transformation class')
        try:
            
            df = self.get_data(store_file_path=self.store_file_path)
            X = df.drop(columns="Quality")

            # replacing -1 with 0 and other kept same as 1
            y = np.where(df['Quality']==-1,0,1)

            logging.info('Splitting data into X_train, X_test, y_train, y_test')
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

            preprocessor = self.get_data_transformation_obj()

            logging.info('Scaling X_train and X_test')
            X_train_scaled = preprocessor.fit_transform(X_train) 
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.preprocessor_object_file
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
            save_object(file_path=preprocessor_path,obj=preprocessor)

            train_arr = np.c_[X_train_scaled,np.array(y_train)]
            test_arr = np.c_[X_test_scaled,np.array(y_test)]

            return (train_arr,
                    test_arr,
                    preprocessor_path)
            
        except Exception as e:
            raise CustomException(e,sys)
        
