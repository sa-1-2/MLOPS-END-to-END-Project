import pandas as pd
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception

import os
import sys
from dataclasses import dataclass

from mlops.utils.utils import load_object


@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e, sys)
