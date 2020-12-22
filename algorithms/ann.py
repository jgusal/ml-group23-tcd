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
    training_data = np.zeros(shape=(462000, 210))

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
    print("loading")
    
    training_data[:, 143:174] = to_categorical(Label_enc.fit_transform(
            [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').day for i in data[0].values]
    ))
    

    training_data[:, 174:176] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').weekday() < 5 for i in data[0].values]
    ))
    

    training_data[:, 176:179] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').month for i in data[0].values]
    ))
    

    training_data[:, 179:186] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').weekday() for i in data[0].values]
    ))

    training_data[:, 186:210] = to_categorical(Label_enc.fit_transform(
        [datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').hour for i in data[0].values]
    ))

    actual_y_value = data.loc[:,19]
    print("loaded")
    return training_data, actual_y_value

    
def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true))) 



training_data, actual_y_value = data()

def model_1():
    model = Sequential()
    model.add(Dense(28, input_shape=(210,), activation='relu'))
    model.add(Dense(28, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer = Adadelta(), loss=root_mean_squared_error, metrics=['mae'])
    return model

def model_2():
    model = Sequential()
    model.add(Dense(28, input_shape=(210,), activation='relu'))
    model.add(Dense(28, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer = Adadelta(), loss=root_mean_squared_error, metrics=['mae'])
    return model


def model_3():
    model = Sequential()
    model.add(Dense(32, input_shape=(210,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer = Adadelta(), loss=root_mean_squared_error, metrics=['mae'])
    return model

def model_4():
    model = Sequential()
    model.add(Dense(64, input_shape=(210,), activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer = Adadelta(), loss=root_mean_squared_error, metrics=['mae'])
    return model

def model_5():
    model = Sequential()
    model.add(Dense(32, input_shape=(210,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer = Adadelta(), loss=root_mean_squared_error, metrics=['mae'])
    return model



from sklearn.model_selection import KFold

models = [model_1, model_2, model_3, model_4, model_5]

best_model = None
best_rmse = None
best_results = None

from sklearn.model_selection import TimeSeriesSplit


for m in models:

    ann_results = []
    for train_index, test_index in TimeSeriesSplit(n_splits=5).split(training_data):
        model = m()
        x_train_data_fold = training_data[train_index[:], :]
        y_train_data_fold = actual_y_value.iloc[train_index[:]]
        x_test_data_fold = training_data[test_index, :]
        y_test_data_fold = actual_y_value.iloc[test_index]

        history = model.fit(
            x_train_data_fold, 
            y_train_data_fold, 
            batch_size=30, 
            epochs=10, 
            validation_split=0.1, 
            callbacks=[keras.callbacks.EarlyStopping(monitor='mae', mode='max', patience=2, restore_best_weights=True)])

        results = model.evaluate(x_test_data_fold, y_test_data_fold)
        ann_results.append(results)
        print(ann_results)

    mean_rmse = statistics.mean([i[0] for i in ann_results])
    mean_mae = statistics.mean([i[1] for i in ann_results])

    if best_model is None or mean_rmse < best_rmse:
        best_model = m
        best_rmse = mean_rmse
        best_results = ann_results



best_mean_rmse = statistics.mean([i[0] for i in best_results])
best_mean_mae = statistics.mean([i[1] for i in best_results])

print("average mean rmse", best_mean_rmse)
print("average mean mae", best_mean_mae)



X_train, X_test, y_train, y_test = train_test_split(training_data, actual_y_value, test_size=0.2, shuffle = False)
model = model_5()

print(X_train)
print(X_test)
model.summary()

history = model.fit(
    X_train, 
    y_train, 
    batch_size=30, 
    epochs=1, 
    validation_split=0.1
    )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Root Mean Squared Error')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()
plt.clf()


plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('ANN Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()
plt.clf()


results = model.evaluate(X_test, y_test)
print(results)
predictions = [i[0] for i in model.predict(X_test)]

from sklearn.metrics import r2_score

model_r2 = r2_score(y_test, predictions)
print("r2 value", model_r2)



kstest_results_actual_values = stats.kstest(y_test, 'norm')
print("kstest_results_actual_values", kstest_results_actual_values)
kstest_results_predicted_values = stats.kstest(predictions, 'norm')
print("kstest_results_predicted_values", kstest_results_predicted_values)

P_VALUE = 0.05

if kstest_results_predicted_values.pvalue <= P_VALUE or kstest_results_predicted_values.pvalue <= P_VALUE:
    print(stats.kruskal(y_test, predictions))
    print(stats.mannwhitneyu(y_test, predictions))
else:
    levene_results = stats.levene(y_test, predictions)
    print("levene_results", levene_results)
    print(stats.ttest_ind(y_test, predictions, equal_var= levene_results.pvalue > P_VALUE))
