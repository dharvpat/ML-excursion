import model
import pre_process
import yfinance as yf
import tensorflow as tf
import numpy as np

ticker = 'AAPL'
data = yf.download(ticker, start='2016-09-01', end='2023-07-01',interval='1d')

num_features = 7 #RSI, Moving average slope, Change in moving average, Signal, MACD, Signal vs MACD, Volume
num_actions = 1 #How many outputs do we want
history_depth = 10 # num of interval points which are accessible to the model before right now
loss_function = tf.keras.losses.MeanSquaredError()

data1 = pre_process.pre_process(data,Moving_window=14)

RSI = data1['RSI'].to_numpy()
ma_slope = data1['Normalized Moving Average Slope'].to_numpy()
MACD = data1['MACD_Value'].to_numpy()
signal = data1['Signal_Value'].to_numpy()
Volume = data1['Volume'].to_numpy()
percentchange = data1['Change'].to_numpy()
percentchange = percentchange[1:]
percentchange = np.append(percentchange,np.NaN)

data_to_input = np.column_stack((RSI,ma_slope,MACD,signal,Volume,percentchange))

training_data = data_to_input[:-20]
validation_data = data_to_input[-20:]

training_data_slices = [training_data.iloc[i:i+history_depth] for i in range(0,len(training_data)-history_depth)]
x_data_train = training_data_slices[:-1]
y_data_train = training_data_slices[-1]

validation_data_slices = [validation_data.iloc[i:i+history_depth] for i in range(0,len(validation_data)-history_depth)]
x_data_val = validation_data_slices[:-1]
y_data_val = validation_data_slices[-1]