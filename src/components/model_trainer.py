import os 
import sys 
import pandas as pd 
import numpy as np 
from typing import Generator, List, Tuple
from src.logger import logging
from src.exception import CustomException
from src.constant import DATABASE_NAME, COLLECTION_NAME,MONGO_DB_URL
from src.utils.utils import save_object

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join('artifact_folder')
    trained_model_path = os.path.join(artifact_folder,'model.pkl')
    expected_accuracy = 0.45
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