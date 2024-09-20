#file used to create code for training model
import os
import sys

from dataclasses import dataclass
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from src.tools.common import save_object, evaluate_models


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass

@dataclass
class TrainingModelConfig:
    trained_model_file_path=os.path.join("outputs","training-model.pkl")

class TrainingModel:
    def __init__(self):
        self.training_model_config=TrainingModelConfig()

    def initiate_training_model(self,training_array,test_array):
        logging.info("initiated training model")
        try:
            logging.info("spliting training and test data")
            X_training,y_training,X_test,y_test=(
                training_array[:,:-1], #all columns except last
                training_array[:,-1],  #all rows except last
                test_array[:,:-1],     #all columns except last
                test_array[:,-1]       #all rows except last
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_models(X_training=X_training,y_training=y_training,X_test=X_test,y_test=y_test,models=models)

            logging.info("evaluating best model name and score")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.5:
                logging.error("unable to find any model with 0.5 and above accuracy,exiting")
                raise CustomException("unable to find any model with 0.5 and above accuracy, exiting")

            logging.info("best model evaluated")
            logging.info("saving trained model objects")
            save_object(
                file_path=self.training_model_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info("completed training model")
            return r2_square
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)