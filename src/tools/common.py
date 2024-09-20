#file used to create code for common functions
import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np

from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#function for saving pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for loading pickle file   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for evaluating models
def evaluate_models(X_training, y_training,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_training, y_training)
            y_training_pred = model.predict(X_training)
            y_test_pred = model.predict(X_test)
            #training_model_score = r2_score(y_training, y_training_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
