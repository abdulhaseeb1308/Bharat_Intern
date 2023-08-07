#!/usr/bin/env python
# coding: utf-8

# In[34]:


#mport the necessary libraries:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
import preprocessing  # Assuming 'preprocessing.py' contains required functions


# In[81]:


#Set a random seed for reproducibility:

np.random.seed(7)


# In[82]:


#Load the dataset from a CSV file ('apple_share_price.csv') containing Apple stock price data and reverse the order of the data:
dataset = pd.read_csv('apple_share_price.csv', usecols=[1, 2, 3, 4])
dataset = dataset.reindex(index=dataset.index[::-1])


# In[42]:


#Create an array obs representing the index for flexibility:
obs = np.arange(1, len(dataset) + 1, 1)


# In[44]:


#Calculate different indicators for prediction:
OHLC_avg = dataset.mean(axis=1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis=1)
close_val = dataset[['Close']]


# In[86]:


#Plot the calculated indicators:
plt.plot(obs, OHLC_avg, 'r', label='OHLC avg')
plt.plot(obs, HLC_avg, 'b', label='HLC avg')
plt.plot(obs, close_val, 'g', label='Closing price')
plt.legend(loc='upper right')
plt.show()


# In[88]:


# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis=1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis=1)
close_val = dataset[['Close']]


# PREPARATION OF TIME SERIES DATASET
if isinstance(OHLC_avg, pd.Series):
    # OHLC_avg is a pandas Series, convert it to a numpy array and reshape
    OHLC_avg = OHLC_avg.values.reshape(-1, 1)  # Convert to 2D array with 1 column
else:
    raise ValueError("OHLC_avg must be a pandas Series.")

scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)


# In[96]:


#Split the data into training and testing sets:
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC, :], OHLC_avg[train_OHLC:len(OHLC_avg), :]


# In[97]:


#Prepare the time series dataset for training and testing the LSTM model:
trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
testX, testY = preprocessing.new_dataset(test_OHLC, 1)


# In[98]:


#Reshape the training and testing data to fit the LSTM model:
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1


# In[99]:


#Build the LSTM model with two LSTM layers and a dense layer with a linear activation function:
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))


# In[100]:


#Compile and train the LSTM model on the training data:
model.compile(loss='mean_squared_error', optimizer='adagrad')  # Try other optimizers like SGD, Adam, etc.
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)


# In[65]:


#Make predictions on both the training and testing sets:
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[101]:


# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL
# ... (code for building the LSTM model)

# MODEL COMPILING AND TRAINING
# ... (code for model compiling and training)

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)

# Reshape trainY to (n_samples, 1) before de-normalizing
trainY = np.reshape(trainY, (-1, 1))
trainY = scaler.inverse_transform(trainY)

testPredict = scaler.inverse_transform(testPredict)

# Reshape testY to (n_samples, 1) before de-normalizing
testY = np.reshape(testY, (-1, 1))
testY = scaler.inverse_transform(testY)





# In[76]:


# Calculate and display the Root Mean Squared Error (RMSE) for both training and testing predictions
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print('Train RMSE: %.2f' % trainScore)

testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print('Test RMSE: %.2f' % testScore)


# In[102]:


#Plot the original OHLC values, training predictions, and testing predictions:
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(OHLC_avg) - 1, :] = testPredict

OHLC_avg = scaler.inverse_transform(OHLC_avg)

plt.plot(OHLC_avg, 'g', label='original dataset')
plt.plot(trainPredictPlot, 'r', label='training set')
plt.plot(testPredictPlot, 'b', label='predicted stock price/test set')
plt.legend(loc='upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of Apple Stocks')
plt.show()


# In[103]:


# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val / last_val
next_val = model.predict(np.reshape(last_val_scaled, (1, 1, 1)))
print("Last Day Value:", last_val.item())
print("Next Day Value:", (last_val * next_val).item())


# In[ ]:




