#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy import stats
import datetime
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split


# In[2]:


def data():
    data = pd.read_csv("merged_data.csv", header=None)
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
    
    labels = data.loc[:,19]
    print("loaded")
    return training_data, labels


# In[7]:


training_data, labels = data()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.25, random_state=11)


# In[10]:


regressor = DecisionTreeRegressor(random_state=0) 


# In[15]:


regressor.fit(X_train, y_train)


# In[16]:


prediction = regressor.predict(X_test)


# In[17]:


corr = (y_test.values == prediction)
t = [e for e in corr if e]
len(t) / len(corr) * 100


# In[18]:


import matplotlib.pyplot as plt


# In[26]:


plt.plot(y_test.values[500: 600])
plt.plot(prediction[500: 600])
plt.show()


# In[23]:


from sklearn.metrics import mean_squared_error


# In[25]:


mean_squared_error(y_test.values, prediction)


# In[ ]:




