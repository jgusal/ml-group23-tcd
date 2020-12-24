import os
os.environ['PYTHONHASHSEED']=str(11)
import matplotlib
import random
random.seed(11)
import numpy as np
np.random.seed(11)
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

from keras.metrics import RootMeanSquaredError as RMSE
import time
import statistics
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.optimizers import Adadelta
import keras
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy import stats
import datetime
from sklearn.metrics import r2_score
import pandas as pd

def data():
    data = pd.read_csv("merged_data.csv", header = None)
    Label_enc = LabelEncoder()
    training_data = np.zeros(shape=(462000, 210))
    training_data[:, 1] = stats.zscore([datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').timestamp() for i in data[0].values])

    training_data[:, 0] = stats.zscore((data.loc[:,2] ))
    
    training_data[:, 2] = stats.zscore((data.loc[:,3]))
    
    training_data[:, 3] = stats.zscore((data.loc[:,4]))
    training_data[:, 4] = stats.zscore((data.loc[:,5]))
    training_data[:, 5] = stats.zscore((data.loc[:,6]))
    training_data[:, 6] = stats.zscore((data.loc[:,7]))
    
    training_data[:, 7] = stats.zscore((data.loc[:,8]))
    training_data[:, 8] = stats.zscore((data.loc[:,9]))
    training_data[:, 9:24] = to_categorical(Label_enc.fit_transform(data.loc[:,10].values))
    training_data[:, 24] = stats.zscore((data.loc[:,11]))
    
    training_data[:, 25] = stats.zscore((data.loc[:,12]))
    training_data[:, 26:136] = to_categorical(Label_enc.fit_transform(data.loc[:,13].values))
    training_data[:, 136:138] = to_categorical(data[14].values)
    training_data[:, 139] = to_categorical(Label_enc.fit_transform(data.loc[:,15].values)).flatten()

    training_data[:, 140:142] = to_categorical(Label_enc.fit_transform(data.loc[:,16].values))
    training_data[:, 142] = stats.zscore((data.loc[:,17]))

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

    actual_y_value = data.iloc[:,19]

    return training_data, actual_y_value

training_data, actual_y_value = data()

split_fraction = 0.715
train_split = int(split_fraction * len(training_data))
step = 4

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10

start_time = time.time()

step = 4

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10

start = past + future
end = start + train_split

sequence_length = int(past / step)

X_train, X_test, y_train, y_test = train_test_split(training_data, actual_y_value, test_size=0.13, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = False)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    X_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    X_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true))) 


for batch in dataset_train.take(1):
    inputs, targets = batch

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics = ["mae"])
    

print(dataset_train)
print("Input shape:", inputs)
print("Target shape:", targets)

history = model.fit(
    dataset_train,
    epochs=20,
    validation_data=dataset_val,
)

end_time = time.time()
time_taken = end_time - start_time

predictions = [i[0] for i in model.predict(tf.compat.v1.data.make_one_shot_iterator(dataset_test))]

test_rmse = mean_squared_error(y_test[:len(predictions)], predictions, squared= False)

test_mae = mean_absolute_error(y_test[:len(predictions)], predictions)
print("Test data MAE: ", test_mae, "\n")

test_mse = mean_squared_error(y_test[:len(predictions)], predictions, squared= False)
test_rmse = np.sqrt(test_mse)
print("Test data RMSE: ", test_rmse, "\n")

test_r2 = r2_score(y_test[:len(predictions)], predictions)
print("Test data R2: ", test_r2, "\n")

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

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN Mean Squared Error')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.clf()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('RNN Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.clf()