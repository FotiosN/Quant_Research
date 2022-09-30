
from pandas_datareader._utils import RemoteDataError
import numpy as np
from datetime import datetime, timedelta,date
import pandas_datareader.data as web
from keras.models import load_model
from tqdm import tqdm
from pickle import load
import yfinance as yf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.optimize as sco
import re
import html2text
import requests, zipfile, io
import warnings
import json
import data_engine as daf
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras
import data_engine as daf
import pandas as pd
import time
from datetime import datetime, timedelta, date
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, QuantileTransformer
from pickle import dump, load
import matplotlib.pyplot as plt
from random import seed
from sklearn.metrics import confusion_matrix,precision_score, accuracy_score
from sklearn.decomposition import PCA
import os

warnings.filterwarnings("ignore")
#GBPUSD=X
#ETH-USD
#REGN

#SPY
stock = 'GBPUSD=X'
#daf.get_earnings_from_Zach(stock)
day_pred = 1

bias = 0.00
days_chart = 2

def get_train_dataset(stock, day_pred):
    days_in_past = 8000
    startdate = datetime.today() - timedelta(days=days_in_past)
    startdate = startdate.strftime("%Y-%m-%d")
    enddate = date.today()
    fed_data = daf.get_fed_date()
    
    
    pricing_data = daf.candle_dataset(stock,day_pred,startdate,enddate)
    final = pricing_data.merge(fed_data,how = 'left', left_on = ['Date'], right_on = ['DATE'] )
    final = final.sort_values('Date', ascending = True)
    final = final.replace([np.inf, -np.inf], np.nan)
    final = final.fillna(method='bfill')
    final = final.fillna(method='ffill')
    del final['DATE']
    return final  

pricing_data = get_train_dataset(stock, day_pred)

##########################################################

pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])

#temp0 = pricing_data
temp0 = pricing_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
X1 = temp0.copy()
del X1['Date'], X1['stock'], X1['subsector']
X1 = X1.loc[:, X1.columns != 'target'].values
Y = temp0.loc[:, temp0.columns == 'target'].values

# norm_scaler = Normalizer()
# X1 = norm_scaler.fit_transform(X1)
# MinMax = MaxAbsScaler()
# X1 = MinMax.fit_transform(X1)
sc = QuantileTransformer()
X1 = sc.fit_transform(X1)

X1 = pd.DataFrame(X1)
X_test = X1.tail(day_pred * days_chart) 
X = X1.iloc[:-day_pred * days_chart , :]
y_test = Y[-(day_pred * days_chart):]
Y = Y[:-(day_pred * days_chart)]

input_layer = len(X.columns.tolist())
print (input_layer)

########################################################################################
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=80)
# model_classifier = Sequential()
# model_classifier.add(Dense(15, input_dim=input_layer, activation='tanh'))
# model_classifier.add(Dense(150, activation='tanh'))
# model_classifier.add(Dense(15, activation='tanh'))
# model_classifier.add(Dense(150, activation='tanh'))
# model_classifier.add(Dense(15, activation='tanh'))
# model_classifier.add(Dense(1, activation='tanh'))
# optimizer = keras.optimizers.Adam(lr=0.000001)

# # fit the keras model on the dataset
# model_classifier.compile(loss='mse', optimizer=optimizer, metrics = 'mse')
# history = model_classifier.fit(X, Y, epochs= 10000, batch_size=200, callbacks=[callback], verbose = 1)
########################################################################################
# save the model
#model_classifier.save(stock +"model_classifier.h5")
model_classifier = load_model(stock +"model_classifier.h5")

y_pred = model_classifier.predict(X_test)

# x = history.history['mse'][-1:5]
# plt.plot(history.history['mse'][-500:])
plt.show()
unseen = (y_test - y_pred)
unseen = unseen * unseen
unseen_mse = sum(unseen)/len(unseen)
print ('real life mse is : ', unseen_mse)
datee = temp0['Date'].tail(day_pred * days_chart)
predicted = pd.DataFrame(y_pred, columns = ['predicted'])
predicted['Date'] = datee.values
predicted["Date"] = predicted["Date"] + timedelta(days=day_pred * 1.448)
predicted['actual'] = y_test
predicted.loc[predicted['actual'] == predicted['actual'] .shift(-1), 'actual'] = np.nan
predicted['actual'].iloc[-1] = np.nan
predicted.set_index('Date', inplace=True)
predicted = predicted.astype(np.float64)
yolo = predicted.copy()
fig = plt.figure(figsize=(15,9))
fig.suptitle(stock, fontsize=40)
plt.plot(yolo['actual'])
plt.plot(yolo['predicted'] - bias)
fig.savefig('Predictions\\' + stock + '.jpg')
plt.show()

last_px = pricing_data['Adj Close'].tail(1)
last_date = pricing_data['Date'].tail(1)

array_length = len(y_pred)
last_element = y_pred[array_length - 1]

print (str(last_date) + '    '  + str(last_px) + '   '+ str(last_px *(1+last_element)))

# df = daf.get_fed_date()
# df.set_index('DATE', inplace=True)

# lista = df.columns.tolist()

# for i in lista:
#     try:
#         fig = plt.figure(figsize=(15,8))
#         fig.suptitle(i, fontsize=40)
#         plt.plot(df[i])
#         plt.show()
#     except:
#         print(i, ' gamithike')
#         continue

# df['SMA_200'] = df['Adj Close'].rolling(200).mean().shift() .astype(float)
# df['EMA_5/EMA_15'] = df['EMA_5']/df['EMA_15']
# df['10day_return'] = df['Adj Close'].pct_change(periods  = 10) .astype(float)




