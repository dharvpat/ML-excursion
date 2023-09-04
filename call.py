import model
import pre_process
import yfinance as yf
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

data = pd.read_csv('~/Desktop/Jordan Belfort/data.csv')
optimizer = Adam(learning_rate=0.1)
num_features = 5 #RSI, Moving average slope, Signal, MACD, Volume
num_actions = 1 #How many outputs do we want
history_depth = 20 # num of interval points which are accessible to the model before right now
loss_function = 'mean_squared_error'
validation_set_number = 20

data1 = pre_process.pre_process(data,Moving_window=20)

percentchange = data1['Change'].to_numpy()
data.drop(['Date','Change'], axis=1, inplace=True)
percentchange = percentchange[1:]
percentchange = np.append(percentchange,0)

for i in range(len(percentchange)):
    if (percentchange[i]>0):
        percentchange[i] = 1
    else:
        percentchange[i] = 0

data_encoded = pd.get_dummies(data1, columns=['Signal_MACD_strength'], dtype = 'float32')
data_encoded['Volume'] = data_encoded['Volume']/100000 
Y = percentchange

X_train, X_test, Y_train, Y_test = train_test_split(data_encoded[:-1],Y[:-1])
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

param_grid = [{'C':np.linspace(93,100,100),'gamma':np.linspace(2,10,100), 'kernel':['rbf'], 'degree': [1]}]
optimal_params = GridSearchCV(SVC(), param_grid, cv =2, scoring='balanced_accuracy',n_jobs=-1, verbose = 2)
optimal_params.fit(X_train_scaled, Y_train)

print(optimal_params.best_params_)
clf_svm = SVC(degree = 1, kernel='rbf', gamma = optimal_params.best_params_['gamma'], C = optimal_params.best_params_['C'], verbose=True)
clf_svm.fit(X_train_scaled, Y_train)
disp=ConfusionMatrixDisplay(confusion_matrix(Y_test, clf_svm.predict(X_test_scaled)))
disp.plot()
plt.show()