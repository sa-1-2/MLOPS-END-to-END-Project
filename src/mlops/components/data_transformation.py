import pandas as pd
import numpy as np
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import os
import sys
from sklearn.pipeline import Pipeline
from mlops.utils.utils import save_object

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationConfig:
    train_path = Path("G:/MLOPS\MLOPS-END-to-END-Project/artifacts/data_ingestion/data/train.csv")
    val_path = Path("G:/MLOPS\MLOPS-END-to-END-Project/artifacts/data_ingestion/data/val.csv")
    test_path = Path("G:/MLOPS\MLOPS-END-to-END-Project/artifacts/data_ingestion/data/test.csv")
    raw_path = "artifacts/data_transformation"
    preprocessor_obj_file_path=os.path.join('artifacts/data_transformation','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor    
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)


    def initialize_data_transformation(self):
        os.makedirs(self.data_transformation_config.raw_path, exist_ok=True)
        try:
            train_df=pd.read_csv(self.data_transformation_config.train_path)
            val_df=pd.read_csv(self.data_transformation_config.val_path)
            test_df = pd.read_csv(self.data_transformation_config.test_path)            
            logging.info("read train, val and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'val Dataframe Head : \n{val_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_val_df=val_df.drop(columns=drop_columns,axis=1)
            target_feature_val_df=val_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=['id'],axis=1)

            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_val_arr=preprocessing_obj.transform(input_feature_val_df)

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training, validation and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            test_arr = input_feature_test_arr

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            return (
                train_arr,
                val_arr,
                test_arr
            )

            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)