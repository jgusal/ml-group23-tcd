from sklearn.metrics import r2_score

import keras
import os
os.environ['PYTHONHASHSEED']=str(11)
import matplotlib
matplotlib.use("TkAgg")
import random
random.seed(11)
import numpy as np
np.random.seed(11)
import tensorflow as tf
tf.random.set_random_seed(11)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys



import statistics
from sklearn.preprocessing import LabelEncoder

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
    
    training_data = np.zeros(shape=(462000, 211))


    training_data[:, 0] = stats.zscore([datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').timestamp() for i in data[0].values])
    training_data[:, 1] = stats.zscore((data.loc[:,2]))
    training_data[:, 2] = stats.zscore((data.loc[:,3]))
    
    training_data[:, 3] = stats.zscore((data.loc[:,4]))
    training_data[:, 4] = stats.zscore((data.loc[:,5]))
    training_data[:, 5] = stats.zscore((data.loc[:,6]))
    training_data[:, 6] = stats.zscore((data.loc[:,7]))
    
    training_data[:, 7] = stats.zscore((data.loc[:,8]))
    training_data[:, 8] = stats.zscore((data.loc[:,9]))
    training_data[:, 9:24] = to_categorical(Label_enc.fit_transform(data[10].values))
    training_data[:, 24] = stats.zscore((data.loc[:,11]))
    
    training_data[:, 25] = stats.zscore((data.loc[:,12]))
    training_data[:, 26:136] = to_categorical(Label_enc.fit_transform(data[13].values))
    training_data[:, 136:138] = to_categorical(data[14].values)
    training_data[:, 139] = to_categorical(Label_enc.fit_transform(data[15].values)).flatten()

    training_data[:, 140:142] = to_categorical(Label_enc.fit_transform(data[16].values))
    training_data[:, 142] = stats.zscore((data.loc[:,17]))
    training_data[:, 143] = stats.zscore((data.loc[:,18]))
    print("loading")
    
    training_data[:, 144:175] = to_categorical(Label_enc.fit_transform(
            [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').day for i in data[0].values]
    ))
    

    training_data[:, 175:177] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').weekday() < 5 for i in data[0].values]
    ))
    

    training_data[:, 177:180] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').month for i in data[0].values]
    ))
    

    training_data[:, 180:187] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').weekday() for i in data[0].values]
    ))

    training_data[:, 187:211] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').hour for i in data[0].values]
    ))

    labels = data.loc[:,19]
    print("loaded")
    return training_data, labels


model = keras.models.load_model('/Users/owner/Desktop/ml_scraper/ann_model.5')
training_data, labels = data()

X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.25, random_state=11)

results = model.predict(X_test)


