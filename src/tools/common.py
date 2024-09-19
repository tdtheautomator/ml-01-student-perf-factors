#file used to create code for common functions
import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np

from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)