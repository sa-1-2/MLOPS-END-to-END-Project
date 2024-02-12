import pandas as pd
import shutil
import zipfile
import urllib.request as request
from mlops.logger.log import logging
from mlops.exception.exception import customexception
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    url = "https://github.com/sa-1-2/MLOPS-END-to-END-Project/raw/main/data.zip"
    zip_path: Path = Path("G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion/data.zip")
    root_dir: Path = Path("G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion")
    unzip_dir: Path = Path("G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion")
    raw_data_path: str = "G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion/data/raw.csv"
    train_data_path: str = "G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion/data/train.csv"
    val_data_path: str = "G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion/data/val.csv"
    test_data_path: str = "G:/MLOPS/MLOPS-END-to-END-Project/artifacts/data_ingestion/test.csv"
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def download_file(self):
        if os.path.exists(self.ingestion_config.root_dir):
            shutil.rmtree(self.ingestion_config.root_dir)

        os.makedirs(self.ingestion_config.root_dir, exist_ok=True)

        if not os.path.exists(self.ingestion_config.zip_path):
            filename, header = request.urlretrieve(url=self.ingestion_config.url, filename=self.ingestion_config.zip_path)
            logging.info(f"{filename} download! with following file info: \n{header}")
        else:
            logging.info(f"File already exists of size: {os.path.getsize(self.ingestion_config.raw_data_path)}")
        
    # function extract_zip_file will extract zip file into directory
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.ingestion_config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(file=self.ingestion_config.zip_path, mode='r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            data = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Reading dataframe")

            logging.info("Starting Train-validation split")
            train_data, val_data = train_test_split(data, test_size=0.25, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=None)
            val_data.to_csv(self.ingestion_config.val_data_path, index=None)

            logging.info("Train-validation split completed")

            os.remove(self.ingestion_config.raw_data_path)


        except Exception as e:
            logging.exception("An error occurred during data ingestion:")
            raise customexception(e, sys)