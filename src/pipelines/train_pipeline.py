import os
import sys 

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            store_file_path = data_ingestion.initiate_data_ingestion()
            return store_file_path 
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,store_file_path):
        try:
            data_transformation = DataTransformation(store_file_path=store_file_path)
            train_arr,test_arr,preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr, preprocessor_path 
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            ) 
            return model_score
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipeline(self):
        try:
            store_file_path = self.start_data_ingestion()
            train_arr, test_arr,preprocessor_path = self.start_data_transformation(store_file_path=store_file_path)
            accuracy_score = self.start_model_training(train_arr,test_arr)
            print("Training completed trained model score: ",accuracy_score)
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = TrainingPipeline()
    obj.run_pipeline()