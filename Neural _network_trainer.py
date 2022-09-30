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
import warnings
import os

warnings.filterwarnings("ignore")


seed(10)

start_perf = time.time()
days_in_past = 10000
days_to_predict_in_the_future = 20
startdate = datetime.today() - timedelta(days=days_in_past)
startdate = startdate.strftime("%Y-%m-%d")
enddate = date.today()
# dfa = pd.read_csv('sec_files\\stock_list_train.csv')
# stocks = dfa['stock'].to_list()
# x = os.path.exists('pricing_data\\fed_data.csv')
# if x == True:
#     fed_data = pd.read_csv('pricing_data\\fed_data.csv')
#     fed_data['DATE'] = pd.to_datetime(fed_data['DATE'])
# else:
#     fed_data = daf.get_fed_date()
#     fed_data.to_csv('pricing_data\\fed_data.csv', index=False)
#     fed_data = pd.read_csv('pricing_data\\fed_data.csv')
#     fed_data['DATE'] = pd.to_datetime(fed_data['DATE'])
# x = os.path.exists('pricing_data\\pricing_data_'+ str(days_to_predict_in_the_future) +'.csv')
# if x == True:
#     pricing_data = pd.read_csv('pricing_data\\pricing_data_'+ str(days_to_predict_in_the_future) +'.csv')
#     pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])
# else:
#     pricing_data = daf.create_weekly_candle_dataset(stocks,days_to_predict_in_the_future,startdate,enddate)
#     pricing_data.to_csv('pricing_data\\pricing_data_'+ str(days_to_predict_in_the_future) +'.csv', index=False)
#     pricing_data = pd.read_csv('pricing_data\\pricing_data_'+ str(days_to_predict_in_the_future) +'.csv')
#     pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])
    

# temp0 = pricing_data.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )
#temp0 = temp0.dropna()

df = pd.read_csv('sec_files\\s&p500.csv')
df = df['list']
df = df.values.tolist()
df = df[400:500]

for stock in df:
#for stock in range(1):#
    try:
        #stock = 'SPCE'#
        day_pred = 20
        bias = -0.1
        days_chart = 1
        #days_inthe_past = 150
        #daf.get_earnings_from_Zach(stock)
        #pricing_data = pd.read_csv('final_data\\'+stock+ str(day_pred)+ '.csv')
        pricing_data = daf.get_train_dataset(stock, day_pred)
        
        
        pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])
        pricing_data = pricing_data.replace([np.inf, -np.inf], np.nan)
        pricing_data = pricing_data.replace([np.inf, np.nan], 0)
        
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
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=80)
        model_classifier = Sequential()
        model_classifier.add(Dense(15, input_dim=input_layer, activation='tanh'))
        model_classifier.add(Dense(150, activation='tanh'))
        model_classifier.add(Dense(15, activation='tanh'))
        model_classifier.add(Dense(150, activation='tanh'))
        model_classifier.add(Dense(15, activation='tanh'))
        model_classifier.add(Dense(1, activation='tanh'))
        optimizer = keras.optimizers.Adam(lr=0.000001)
        
        # fit the keras model on the dataset
        model_classifier.compile(loss='mse', optimizer=optimizer, metrics = 'mse')
        history = model_classifier.fit(X, Y, epochs= 50000, batch_size=300, callbacks=[callback], verbose = 1)
        ########################################################################################
        # save the model
        model_classifier.save('models\\' + stock +'_' + str(day_pred) + '_' +"model_classifier.h5")
        #model_classifier = load_model(stock +"model_classifier.h5")
        y_pred = model_classifier.predict(X_test)
        
        
        x = history.history['mse'][-1:5]
        plt.plot(history.history['mse'][-500:])
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
        fig.savefig('Predictions\\' + stock + '_'+ str(day_pred) + '.jpg')
        plt.show()

        
        print ('*************Correlation is '+ str(yolo['actual'].corr(yolo['predicted'])) +'***************')
        
        print ("***************Grouped by predicted returns / actual returns*****************")
        #print 'actual'/'predicted' grouped returns
        banana = yolo[['actual','predicted']]
        banana = banana.groupby(pd.cut(banana["predicted"]-bias, np.arange(-1, 1, 0.01))).mean()
        plt.plot(banana['predicted'], banana['actual'])
        plt.show()
        
        
        print ("***************Distributions Actual vs Predicted*****************")
        #print distribution
        banana = yolo[['actual','predicted']]
        ax = banana.plot.kde()
        ax.set_xlim(-0.5,0.5)
        
        print ("***************BINARY CLASSIFICATION - EVALUATION*****************")
        print ("________________________________________________________________")
        matrix = yolo.copy()
        matrix['actual'] = np.where(matrix['actual'] > 0, 1,0)
        matrix['predicted'] = np.where(matrix['predicted'] - bias > 0, 1,0)
        c_matrix = pd.DataFrame(confusion_matrix(matrix['actual'], matrix['predicted']))
        print ("confusion_matrix")
        print (c_matrix)
        print ("________________________________________________________________")
        print ("precision_score ", precision_score(matrix['actual'], matrix['predicted']))
        print ("________________________________________________________________")
        print ("accuracy ", accuracy_score(matrix['actual'], matrix['predicted']))
        print ("________________________________________________________________")
        #if yolo['actual'].corr(yolo['predicted']) >= 0.0:
        yolo.to_csv('results\\' + stock + '_'+ str(day_pred)+ '_' + str(yolo['actual'].corr(yolo['predicted'])) + '_results.csv', index=False)  
    except:
        continue    