import pandas as pd
import os
import sys
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception
from mlops.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dataclasses import dataclass

from mlops.utils.utils import load_object


@dataclass
class ModelEvaluationConfig:
    model_path = os.path.join("artifacts/model_trainer","model.pkl")

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()
        logging.info("Evaluation started")

    def eval_metrics(self, actual, pred):
        rmse = mean_squared_error(actual, pred, squared=False)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, val_array):
        try:
            X_val, y_val = (val_array[:,:-1], val_array[:,-1])
            
            model = load_object(self.model_evaluation_config.model_path)

            #mlflow.set_registry_uri("")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_val)

                rmse, mae, r2 = self.eval_metrics(y_val, prediction)
                mlflow.log_metrics({"rmse":rmse, "mae":mae, "r2":r2})
                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logging.info()
            raise customexception(e, sys)
