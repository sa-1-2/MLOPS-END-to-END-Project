import pandas as pd
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import os
import sys
from sklearn.pipeline import Pipeline
from mlops.utils import save_object

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e, sys)
