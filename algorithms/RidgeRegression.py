seed = 13
import os
os.environ['PYTHONHASHSEED']=str(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import pandas as pd
from scipy import stats
import datetime
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import time


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
    train_size = int(len(full_data) * 0.75)
    historic, future = full_data[0:train_size], full_data[train_size:len(full_data)]

    historic_target = historic.available_bikes
    future_real_target = future.available_bikes

    historic_train = historic.drop(['available_bikes'], axis=1)
    future_test = future.drop(['available_bikes'], axis=1)

    return historic_train, historic_target, future_test, future_real_target


## Cross-validation
def cross_val(historic_train, historic_target, future_test, future_real_target):

    # Cross validation for ridge regression alpha parameter
    model = Ridge()
    param_grid = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=5)
    gsearch = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', return_train_score=True, n_jobs = -1)
    gsearch.fit(historic_train, historic_target)

    results = gsearch.cv_results_
    return results


csv = "../data/transform_data/merged_data.csv"
data = get_data(csv)
historic_train, historic_target, future_test, future_real_target = split_data(data)
results = cross_val(historic_train, historic_target, future_test, future_real_target)

train_mse = results['mean_train_score']
test_mse = results['mean_test_score']
alpha = results['param_alpha']

# Plot alpha against MSE
plt.figure(figsize=(10, 7))
plt.plot(np.log10(alpha.astype(float)), np.abs(train_mse), label = 'Train')
plt.plot(np.log10(alpha.astype(float)), np.abs(test_mse), label = 'Test')
plt.xlabel('Alpha (log10)')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Final model
start_time = time.time()
model = Ridge(alpha = 100)
model.fit(historic_train, historic_target)

# Predictions on historic data
y_train_pred = model.predict(historic_train)

# Predictions on future data
y_future_pred = model.predict(future_test)
end_time = time.time()
time_taken = end_time - start_time


## Evaluation
# MAE for training data
train_mae = mean_absolute_error(historic_target, y_train_pred)
print("Training data MAE: ", train_mae, "\n")

# MAE for test data
test_mae = mean_absolute_error(future_real_target, y_future_pred)
print("Test data MAE: ", test_mae, "\n")

# RMSE
test_mse = mean_squared_error(future_real_target, y_future_pred)
test_rmse = np.sqrt(test_mse)
print("Test data RMSE: ", test_rmse, "\n")

# R-Squared
test_r2 = r2_score(future_real_target, y_future_pred)
print("Test data R2: ", test_r2, "\n")


## Statistics
kstest_results_actual_values = stats.kstest(future_real_target, 'norm')
print("kstest_results_actual_values", kstest_results_actual_values)
kstest_results_predicted_values = stats.kstest(y_future_pred, 'norm')
print("kstest_results_predicted_values", kstest_results_predicted_values)

P_VALUE = 0.05

if kstest_results_predicted_values.pvalue <= P_VALUE or kstest_results_predicted_values.pvalue <= P_VALUE:
    print(stats.kruskal(future_real_target, y_future_pred))
    print(stats.mannwhitneyu(future_real_target, y_future_pred))
else:
    levene_results = stats.levene(future_real_target, y_future_pred)
    print("levene_results", levene_results)
    print(stats.ttest_ind(future_real_target, y_future_pred, equal_var= levene_results.pvalue > P_VALUE))


## Export results
n_features = future_test.shape[1]

res = {'MAE': test_mae,
       'RMSE': test_rmse,
       'R2': test_r2,
       'Num Features': n_features,
       'Time Taken': time_taken
      }

res_df = pd.DataFrame([res], columns = ['MAE', 'RMSE', 'R2', 'Num Features', 'Time Taken'])
result = res_df.to_string()

print(result,  file=open('../algorithms/results/RidgeRegression.txt', 'w'))
