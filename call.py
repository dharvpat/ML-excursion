import model
import pre_process
import yfinance as yf
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam
import pandas as pd

data = pd.read_csv('data.csv')

optimizer = RMSprop(learning_rate=20, momentum=0.9)
num_features = 5 #RSI, Moving average slope, Signal, MACD, Volume
num_actions = 1 #How many outputs do we want
history_depth = 50 # num of interval points which are accessible to the model before right now
loss_function = 'mean_squared_error'
validation_set_number = 20

data1 = pre_process.pre_process(data,Moving_window=14)

RSI = data1['RSI'].to_numpy()
ma_slope = data1['Normalized Moving Average Slope'].to_numpy()
MACD = data1['MACD_Value'].to_numpy()
signal = data1['Signal_Value'].to_numpy()
Volume = data1['Volume'].to_numpy()
percentchange = data1['Change'].to_numpy()
MACD_strength = data1['Signal_MACD_strength'].to_numpy()
percentchange = percentchange[1:]
percentchange = np.append(percentchange,np.NaN)
data_to_input = np.column_stack((RSI,Volume,MACD_strength,MACD,signal,percentchange))

training_data = data_to_input[:-validation_set_number*history_depth]
validation_data = data_to_input[-validation_set_number*history_depth:]

training_data_slices = np.array([training_data[i:i+history_depth] for i in range(0,len(training_data)-history_depth)])
x_data_train = np.array(training_data_slices[:,:,:-1])
y_data_train = np.array(training_data_slices[:,:,-1])

validation_data_slices = np.array([validation_data[i:i+history_depth] for i in range(0,len(validation_data)-history_depth)])
x_data_val = np.array(validation_data_slices[:,:,:-1])
y_data_val = np.array(validation_data_slices[:,:,-1])

training_labels = []
for i in range(len(y_data_train)):
    if (y_data_train[i,history_depth-1] > 0):
        training_labels.append(1)
    else:
        training_labels.append(-1)

testing_labels = []
for i in range(len(y_data_val)):
    if (y_data_val[i,history_depth-1] > 0):
        testing_labels.append(1)
    else:
        testing_labels.append(-1)

y_data_train = np.array(training_labels)
y_data_val = np.array(testing_labels)
model_object = model.model(num_features=num_features, num_actions=num_actions, history_depth=history_depth, loss_function=loss_function, optimizer=optimizer)
history = model_object.fit(x_data_train,y_data_train,epochs = 1000, validation_data=(x_data_val,y_data_val))