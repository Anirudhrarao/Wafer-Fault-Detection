import os 
import sys 
import pandas as pd 
import numpy as np 
from typing import Generator, List, Tuple
from src.logger import logging
from src.exception import CustomException
from src.constant import DATABASE_NAME, COLLECTION_NAME,MONGO_DB_URL
from src.utils.utils import save_object, read_yaml_file

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join('artifact')
    trained_model_path = os.path.join(artifact_folder,'model.pkl')
    expected_accuracy = 0.5
    model_config_file_path = os.path.join('config','model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            'XGBClassifier':XGBClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'SVC':SVC(),
            'RandomForestClassifier':RandomForestClassifier()
        }


    def evaluate_model(self,X,y,models:dict):
        '''
            Desc: This function will train different different model and store accuracy score
            report dict
        '''
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train,y_train)
                # Predicting X_train
                y_train_pred = model.predict(X_train)
                # Predicting X_test
                y_test_pred = model.predict(X_test)
                train_model_score = accuracy_score(y_train,y_train_pred)
                test_model_score = accuracy_score(y_test,y_test_pred)
                report[list(models.keys())[i]] = test_model_score
            return report
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_best_model(self,
                    x_train:np.array, 
                    y_train: np.array,
                    x_test:np.array, 
                    y_test: np.array):
        
        '''
        Desc: This will compare model and will give best model with high accuracy
        '''
        try:
            model_report: dict = self.evaluate_model(
                x_train =  x_train, 
                y_train = y_train, 
                x_test =  x_test, 
                y_test = y_test, 
                models = self.models
            )
            print(model_report)
            best_model_score = max(sorted(model_report.values()))
            # To get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_obj = self.models[best_model_name]

            return (
                best_model_score,
                best_model_name,
                best_model_obj
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def hyper_parameter_tune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object:
        '''
            Desc: This function will responsible for hyperparameter tunning
        '''
        try:
            model_params_grid = read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            # grid search cv
            grid_search = GridSearchCV(best_model_object,param_grid=model_params_grid,cv=5,n_jobs=-1,verbose=1)
            grid_search.fit(X_train,y_train)
            # To get best params
            best_params = grid_search.best_params_
            logging.info(f"We got the best params: {best_params} for model")
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_trainer(self,train_array,test_array):
        '''
            Desc: This method initiate model trainer class
        '''
        try:
            logging.info('Splitting training and testing input and target feature')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            logging.info(f"Extracting model config file path")
            model_report: dict = self.evaluate_model(X=x_train,y=y_train,models=self.models)
            # To get best model score from model_report dict
            best_model_score = max(sorted(model_report.values()))
            # To get best model name from model_report dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = self.models[best_model_name]
            best_model = self.hyper_parameter_tune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )
            best_model.fit(x_train,y_train)
            # predicting
            y_pred = best_model.predict(x_test)

            best_model_score = accuracy_score(y_test,y_pred)
            print(f'Found best model: {best_model_name} with accuracy score: {best_model_score}')

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception('No model found with accuracy greater than threshold 0.5')
            
            logging.info(f'Found best model: {best_model_name} with accuracy score: {best_model_score}')
            logging.info(f"Saving model pickle file {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path),exist_ok=True)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path
        
        except Exception as e:
            raise CustomException(e,sys)