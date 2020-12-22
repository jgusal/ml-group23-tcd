import os
os.environ['PYTHONHASHSEED']=str(11)
import random
random.seed(11)
import numpy as np
np.random.seed(11)

import statistics

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from scipy import stats
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def data():
    data = pd.read_csv("data/transform_data/merged_data.csv", header=None)
    training_data = np.zeros(shape=(462000, 210))

    actual_y_value = data.loc[:,19]
    return training_data, actual_y_value


training_data, actual_y_value = data()

X_train, X_test, y_train, y_test = train_test_split(training_data, actual_y_value, test_size=0.13, shuffle = False)


def mean_baseline(X_train, y_train):
    return statistics.mean(y_train)

mean_available_bikes = mean_baseline(X_train, y_train)

mean_predictions = [mean_available_bikes for i in y_test]

test_rmse = mean_squared_error(y_test, mean_predictions, squared= False)

test_mae = mean_absolute_error(y_test, mean_predictions)
print("Test data MAE: ", test_mae, "\n")

test_mse = mean_squared_error(y_test, mean_predictions, squared= False)
test_rmse = np.sqrt(test_mse)
print("Test data RMSE: ", test_rmse, "\n")

test_r2 = r2_score(y_test, mean_predictions)
print("Test data R2: ", test_r2, "\n")


