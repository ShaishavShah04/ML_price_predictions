#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas_datareader as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


symbol = 'FB'


# In[3]:


# Getting stock Data
df = pdr.DataReader(symbol, data_source = 'yahoo', start= '2012-01-01', end='2021-01-20')


# In[4]:


df.shape


# In[5]:


#Visualizing Closing History
plt.figure(figsize=(16,8))
plt.title('Close Price History SPY')
plt.plot(df['Close'])
plt.xlabel('Data')
plt.ylabel('USD Closing Prices')
# In[6]:


#Create a new df with only closing prices
close_df = df.filter(['Close'])
#Going to np
dataset = close_df.values
# Num of rows to train data
training_data_len = math.ceil(len(dataset)*0.8)


# In[7]:


# Scaling the Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# In[8]:


# Create training Dataset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


# In[9]:


#Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[10]:


#Reshape the xtrain data
# We need 3d data, not 2d
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[11]:


# Building model here
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[12]:


#Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[13]:


#Train Models
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[14]:


# Creating testing data
test_data = scaled_data[training_data_len - 60:,:]
#Create the datsets x_test, y_test
x_test= []
y_test=dataset[training_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[15]:


#Covert Data into numpy
x_test = np.array(x_test)


# In[16]:


#Reshaping
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[17]:


#Predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[18]:


# getting the RMSE
rmse = np.sqrt(np.mean(predictions-y_test)**2)
rmse


# In[19]:


train = close_df[:training_data_len]
valid = close_df[training_data_len:]
valid['Predictions'] = predictions
# graphs
plt.figure(figsize=(16,8))
plt.title(f'Model Predictions for {symbol}')
plt.xlabel('Dates',fontsize=12)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()

### Trying to predict the price for future Date (one day after the end date of data)
stock_quote = pdr.DataReader(symbol, data_source='yahoo', start='2012-01-01', end='2021-01-28')
new_df = stock_quote.filter(['Close'])
last_60_days = new_df[-60:].values
# Scale data
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # 3d now
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)



