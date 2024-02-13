import os
import sys
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import pandas as pd
from mlops.components.data_ingestion import DataIngestion
from mlops.components.data_transformation import DataTransformation
from mlops.components.model_trainer import ModelTrainer
from mlops.components.model_evaluation import ModelEvaluation

logging.info("Data Ingestion started")
data_ingest = DataIngestion()
data_ingest.download_file()
logging.info("File Downloaded Successfully")
data_ingest.extract_zip_file()
logging.info("Datasets extracted Successfully")
data_ingest.initiate_data_ingestion()
logging.info("Data Ingestion completed")
logging.info("Data Transformation Started")
data_transform = DataTransformation()
train_arr, val_arr, test_arr = data_transform.initialize_data_transformation()
logging.info("Data Transformation Completed")
logging.info("Model Training Started")
model_trainer = ModelTrainer()
model_trainer.initiate_model_trainer(train_array=train_arr, val_array=val_arr)
logging.info("Model Training Finished")
logging.info("Model Evaluation Started")
model_evaluation = ModelEvaluation()
model_evaluation.initiate_model_evaluation(val_arr)
logging.info("Model Evaluation Competed")