import pandas as pd
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from mlops.utils.utils import savae_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    pass

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e, sys)
