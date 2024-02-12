import pandas as pd
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
import pickle

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from mlops.utils.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts/model_trainer', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, val_array):
        try:
            logging.info("Splitting Dependent & Independent variables")
            X_train, X_val, y_train, y_val = (train_array[:,:-1], 
                                              val_array[:,:-1],
                                              train_array[:,-1],
                                              val_array[:,-1]
                                              )
            models= { 
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'RandomForestRegressor':RandomForestRegressor(),
                'XGBRegressor':XGBRegressor()
                }
            model_report = evaluate_model(X_train, y_train, X_val, y_val, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Error Occured in model training")
            raise customexception(e, sys)
