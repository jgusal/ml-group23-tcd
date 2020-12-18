import statistics
from sklearn.preprocessing import LabelEncoder

labels_index = 19

import random
import json
import sys
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import keras
import pandas as pd
from hyperas.distributions import uniform
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import plot_model




random.seed(42)
np.random.seed(seed=42)
from sklearn.preprocessing import LabelEncoder

from scipy import stats
import datetime

def data():
    data = pd.read_csv("data/transform_data/merged_data.csv", header=None)
    Label_enc = LabelEncoder()
    
    data[0] = stats.zscore([datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').timestamp() for i in data[0].values])
    data[1] = Label_enc.fit_transform(data[1].values)
    data[2] = stats.zscore((data.loc[:,2]))
    data[3] = stats.zscore((data.loc[:,3]))
    data[4] = stats.zscore((data.loc[:,4]))
    data[5] = stats.zscore((data.loc[:,5]))
    data[6] = stats.zscore((data.loc[:,6]))
    data[7] = stats.zscore((data.loc[:,7]))
    data[8] = stats.zscore((data.loc[:,8]))
    data[9] = stats.zscore((data.loc[:,9]))
    data[10] = Label_enc.fit_transform(data[10].values)
    data[11] = stats.zscore((data.loc[:,11]))
    data[12] = stats.zscore((data.loc[:,12]))
    data[13] = Label_enc.fit_transform(data[13].values)
    
    data[14] = Label_enc.fit_transform(data[14].values)
    data[15] = Label_enc.fit_transform(data[15].values)
    data[16] = Label_enc.fit_transform(data[16].values)

    data[17] = stats.zscore((data.loc[:,17]))
    data[18] = stats.zscore((data.loc[:,18]))
    data[19] = stats.zscore((data.loc[:,19]))
    
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:18], data.iloc[:,19], test_size=0.25, random_state=42)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = data()

model = Sequential()
model.add(Dense(16, input_shape=(18,), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer = RMSprop(), loss='mse', metrics=['mae'])

model.fit(X_train, y_train, batch_size=5, epochs=30, callbacks=[keras.callbacks.EarlyStopping(monitor='acc', mode='max', patience=2, restore_best_weights=True)])
results = model.evaluate(X_test, y_test)
print(results)

