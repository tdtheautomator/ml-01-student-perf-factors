#file used to create code for ingesting data
import os
import sys
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('outputs',"raw_data.csv")
    training_data_path: str=os.path.join('outputs',"training_data.csv")
    test_data_path: str=os.path.join('outputs',"test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("initiated data ingestion")
        try:
            #curr_dir=os.getcwd()
            #logging.info(f'current directory is {curr_dir}')
            logging.info('reading datset as dataframe')
            df=pd.read_csv('./data/student_performance_factors.csv')
            
            logging.info("creating outputs folders if doesn't exist")
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path),exist_ok=True)
            
            logging.info('exporting dataframe to csv to outputs')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("spliting training and test data")
            training_set,test_set=train_test_split(df,test_size=0.2,random_state=None)
            
            logging.info('exporting training dataset to outputs')
            training_set.to_csv(self.ingestion_config.training_data_path,index=False,header=True)
            
            logging.info('exporting test dataset to outputs')
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion completed")

            return(
                self.ingestion_config.training_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    training_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    training_arr,test_arr,_=data_transformation.initiate_data_transformation(training_data,test_data)
