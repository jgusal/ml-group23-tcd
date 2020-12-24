from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy import stats
import datetime
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import tree
from scipy import stats


## Encode data
def get_data(csv):
    data = pd.read_csv(csv, header=None)

    data[0] = stats.zscore([datetime.datetime.strptime(i, '%Y-%m-%d_%H-%M-%S').timestamp() for i in data[0].values])
    data[2] = stats.zscore((data.loc[:,2]))
    data[3] = stats.zscore((data.loc[:,3]))
    data[4] = stats.zscore((data.loc[:,4]))
    data[5] = stats.zscore((data.loc[:,5]))
    data[6] = stats.zscore((data.loc[:,6]))
    data[7] = stats.zscore((data.loc[:,7]))
    data[8] = stats.zscore((data.loc[:,8]))
    data[9] = stats.zscore((data.loc[:,9]))
    data[11] = stats.zscore((data.loc[:,11]))
    data[12] = stats.zscore((data.loc[:,12]))
    data[17] = stats.zscore((data.loc[:,17]))
    data[18] = stats.zscore((data.loc[:,18]))
    #data[19] = stats.zscore((data.loc[:,19]))

    data.columns= ['timestamp', 'city', 'av_temp', 'min_temp', 'max_temp', 'pressure',
                   'humidity', 'visibility', 'wind_speed', 'clouds', 'weather_desc',
                   'sunrise', 'sundown', 'bike_station_address', 'banking', 'bonus',
                   'status', 'total_bike_stands', 'available_bike_stands', 'available_bikes']

    dummy = pd.get_dummies(data, prefix=['city', 'weather_desc', 'bike_station_address', 'banking', 'bonus', 'status'],
                           columns=['city', 'weather_desc', 'bike_station_address', 'banking', 'bonus', 'status'])

    final = dummy.drop(['banking_False', 'status_False', 'available_bike_stands'], axis=1)
    final = final.fillna(final.mean())

    return final


## Split data
def split_data(full_data):

    # Split data into historic and future data to evaluate our future predictions
    train_size = int(len(full_data) * 0.87)
    historic, future = full_data[0:train_size], full_data[train_size:len(full_data)]

    historic_target = historic.available_bikes
    future_real_target = future.available_bikes

    historic_train = historic.drop(['available_bikes'], axis=1)
    future_test = future.drop(['available_bikes'], axis=1)

    return historic_train, historic_target, future_test, future_real_target

# Read data
csv = "../data/transformed_data/merged_data.csv"
data = get_data(csv)
historic_train, historic_target, future_test, future_real_target = split_data(data)
X_train, y_train, X_test, y_test = (historic_train, historic_target, future_test, future_real_target)


# Final model
regressor = DecisionTreeRegressor(random_state=0) 

start = time.time()
regressor.fit(X_train, y_train)
end = time.time()

print("Training took %f seconds" % (end - start))

# Predict test data
prediction = regressor.predict(X_test)

## MAE for training data
train_mae = mean_absolute_error(y_train, regressor.predict(X_train))
print("Training data MAE: ", train_mae, "\n")

## MAE for test data
test_mae = mean_absolute_error(y_test, prediction)
print("Test data MAE: ", test_mae, "\n")

## RMSE
test_mse = mean_squared_error(y_test, prediction)
test_rmse = np.sqrt(test_mse)
print("Test data MSE: ", test_mse)
print("Test data RMSE: ", test_rmse, "\n")

## R-Squared
test_r2 = r2_score(y_test, prediction)
print("Test data R2: ", test_r2, "\n")

# Plot first levels of the decision tree
tree.plot_tree(regressor, max_depth = 2)
plt.show()

## Statistics
kstest_results_actual_values = stats.kstest(future_real_target, 'norm')
print("kstest_results_actual_values", kstest_results_actual_values)
kstest_results_predicted_values = stats.kstest(prediction, 'norm')
print("kstest_results_predicted_values", kstest_results_predicted_values)

P_VALUE = 0.05

if kstest_results_predicted_values.pvalue <= P_VALUE or kstest_results_predicted_values.pvalue <= P_VALUE:
    print(stats.kruskal(future_real_target, prediction))
    print(stats.mannwhitneyu(future_real_target, prediction))
else:
    levene_results = stats.levene(future_real_target, prediction)
    print("levene_results", levene_results)
    print(stats.ttest_ind(future_real_target, prediction, equal_var= levene_results.pvalue > P_VALUE))