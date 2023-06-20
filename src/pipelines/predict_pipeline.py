import os
import sys
import pickle
import pandas as pd
from flask import request
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object

from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname:str = "predictions"
    prediction_file_name:str = "predicted_file.csv"
    model_file_path:str = os.path.join('artifact','model.pkl')
    preprocessor_path:str = os.path.join('artifact','preprocessor.pkl')
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)


class PredictionPipeline:
    def __init__(self):
        self.request = request
        self.prediction_pipeline_config = PredictionPipelineConfig()

    
    def save_input_file(self)->str:
        '''
            Desc: This method saves the input file to the prediction artifacts directory. 
        '''
        try:
            file_input = 'prediction_artifacts'
            os.makedirs(file_input,exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(file_input,input_csv_file.filename)

            input_csv_file.save(pred_file_path)

            return pred_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self,feature):
        '''
            Desc: This function responsible for prediction  
        '''
        try:
            model = load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = load_object(self.prediction_pipeline_config.preprocessor_path)
            transformed_x = preprocessor.transform(feature)
            predictions = model.predict(transformed_x)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_predicted_df(self,input_df:pd.DataFrame):
        '''
            Desc: This method returns the data frame with a new column containing predictions
        '''
        try:
            prediction_column_name:str = 'Quality'
            input_dataframe:pd.DataFrame = pd.read_csv(input_df)
            input_dataframe = input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe else input_dataframe

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad',1:'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")

        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_file()
            self.get_predicted_df(input_csv_path)

            return self.prediction_pipeline_config

        except Exception as e:
            raise CustomException(e,sys)
