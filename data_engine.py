
from pandas_datareader._utils import RemoteDataError
import numpy as np
from datetime import datetime, timedelta,date
import pandas_datareader.data as web
from keras.models import load_model
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
import os
import data_engine as daf
from sklearn.preprocessing import  QuantileTransformer
from random import seed
import matplotlib as plt

def warn(*args, **kwargs):
    pass

warnings.filterwarnings("ignore")

def get_pred(stock,day_pred,bias,days_chart):
    seed(10)
    days_in_past = 10000
    startdate = datetime.today() - timedelta(days=days_in_past)
    startdate = startdate.strftime("%Y-%m-%d")
    try:
        pricing_data = daf.get_train_dataset(stock, day_pred)
        pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])
        pricing_data = pricing_data.replace([np.inf, -np.inf], np.nan)
        pricing_data = pricing_data.replace([np.inf, np.nan], 0)
        temp0 = pricing_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X1 = temp0.copy()
        del X1['Date'], X1['stock'], X1['subsector']
        X1 = X1.loc[:, X1.columns != 'target'].values
        Y = temp0.loc[:, temp0.columns == 'target'].values
        sc = QuantileTransformer()
        X1 = sc.fit_transform(X1)
        X1 = pd.DataFrame(X1)
        X_test = X1.tail(day_pred * days_chart)
        y_test = Y[-(day_pred * days_chart):]
        Y = Y[:-(day_pred * days_chart)]
        model_classifier = load_model('models\\' + stock +'_' + str(day_pred) + '_' +"model_classifier.h5")
        y_pred = model_classifier.predict(X_test)
        unseen = (y_test - y_pred)
        unseen = unseen * unseen
        unseen_mse = sum(unseen)/len(unseen)
        print ('real life mse is : ', unseen_mse)
        datee = temp0['Date'].tail(day_pred * days_chart)
        predicted = pd.DataFrame(y_pred + bias, columns = ['predicted'])
        predicted['Date'] = datee.values
        predicted["Date"] = predicted["Date"] + timedelta(days=day_pred * 1.448)
        predicted['actual'] = y_test
        predicted.loc[predicted['actual'] == predicted['actual'] .shift(-1), 'actual'] = np.nan
        predicted['actual'].iloc[-1] = np.nan
        predicted.set_index('Date', inplace=True)
        predicted = predicted.astype(np.float64)
        yolo = predicted.copy()
        # fig = plt.figure(figsize=(15,9))
        # fig.suptitle(stock, fontsize=40)
        # plt.plot(yolo['actual'])
        # plt.plot(yolo['predicted'] + bias)
        # fig.savefig('Predictions\\' + stock + '_'+ str(day_pred) + '.jpg')
        # plt.show()
        # banana = yolo[['actual','predicted']]
        # banana = banana.groupby(pd.cut(banana["predicted"] + bias, np.arange(-1, 1, 0.01))).mean()
        # plt.plot(banana['predicted'], banana['actual'])
        # plt.show()
        # banana = yolo[['actual','predicted']]
        # matrix = yolo.copy()
        # matrix['actual'] = np.where(matrix['actual'] > 0, 1,0)
        # matrix['predicted'] = np.where(matrix['predicted'] + bias > 0, 1,0)
        #if yolo['actual'].corr(yolo['predicted']) >= 0.0:
        yolo.to_csv('results\\' + stock + '_'+ str(day_pred)+ '_' + str(yolo['actual'].corr(yolo['predicted'])) + '_results.csv', index=False) 
        return yolo
    except:
        print (stock + ' not valid')    
        
        

  
    
def beta(individual, market, period): 
    returns = individual.join(market).dropna()
    returns = returns.pct_change().dropna()
    cov = returns.iloc[0:,0].rolling(period).cov(returns.iloc[0:,1])
    market_var = returns.iloc[0:,1].rolling(period).var()
    individual_beta = cov / market_var
    return individual_beta

def get_train_dataset(stock, days_to_pred):
    benchmark = yf.download('SPY')
    benchmark.reset_index(level=0, inplace=True)
    benchmark = benchmark[['Date','Adj Close']]
    benchmark = benchmark.rename(columns={'Adj Close': 'BM px'})
    
    
    fed_data = get_fed_date()
    
    
    #fed_data = pd.read_csv('pricing_data\\fed_data.csv')
    #fed_data['DATE'] = pd.to_datetime(fed_data['DATE'], errors='coerce')
    
    
    days_in_past = 9000
    startdate = datetime.today() - timedelta(days=days_in_past)
    startdate = startdate.strftime("%Y-%m-%d")
    enddate = date.today()
    
    
    earnings_from_Zach = pd.read_csv('Earnings\\'+ stock +'_earnings.csv')
    Balance_sheet1 = get_balace_sheet(stock)
    Balance_sheet1['Date'] = pd.to_datetime(Balance_sheet1['Date'], errors='coerce')
    #earnings_from_Zach = daf.get_earnings_from_Zach(stock)
    
    
    earnings_from_Zach = pd.read_csv('Earnings\\'+ stock +'_earnings.csv')
    earnings_from_Zach['Date'] = pd.to_datetime(earnings_from_Zach['Date'], errors='coerce')
    
    
    pricing_data = create_weekly_candle_dataset([stock],days_to_pred,startdate,enddate)
    
    
    pricing_data = pricing_data.merge(benchmark,how = 'inner', left_on = ['Date'], right_on = ['Date'] )
    
    
    pricing_data['beta_monthly'] = beta(pricing_data['Adj Close'].to_frame(), pricing_data['BM px'].to_frame(), 22)
    pricing_data['beta_1y'] = beta(pricing_data['Adj Close'].to_frame(), pricing_data['BM px'].to_frame(), 252)
    pricing_data['beta_3y'] = beta(pricing_data['Adj Close'].to_frame(), pricing_data['BM px'].to_frame(), 756)
    
    pricing_data = pricing_data.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )
    
    
    pricing_data['T10Y2Y_3y_spread'] = pricing_data['T10Y2Y'].rolling(750).mean().shift().astype(float)/pricing_data['T10Y2Y']
    pricing_data['T10Y2Y_y_spread'] = pricing_data['T10Y2Y'].rolling(250).mean().shift().astype(float)/pricing_data['T10Y2Y']
    pricing_data['T10Y2Y_sa_spread'] = pricing_data['T10Y2Y'].rolling(90).mean().shift().astype(float)/pricing_data['T10Y2Y']
    pricing_data['T10Y2Y_m_spread'] = pricing_data['T10Y2Y'].rolling(30).mean().shift().astype(float)/pricing_data['T10Y2Y']
    pricing_data['T10Y2Y_6monthly_return'] = pricing_data['T10Y2Y'].pct_change(periods  = 176) .astype(float)
    pricing_data['T10Y2Y_12monthly_return'] = pricing_data['T10Y2Y'].pct_change(periods  = 252) .astype(float)
    pricing_data['T10Y2Y_36monthly_return'] = pricing_data['T10Y2Y'].pct_change(periods  = 500) .astype(float)
    
    pricing_data['T5YIFR_3y_spread'] = pricing_data['T5YIFR'].rolling(750).mean().shift().astype(float)/pricing_data['T5YIFR']
    pricing_data['T5YIFR_y_spread'] = pricing_data['T5YIFR'].rolling(250).mean().shift().astype(float)/pricing_data['T5YIFR']
    pricing_data['T5YIFR_sa_spread'] = pricing_data['T5YIFR'].rolling(90).mean().shift().astype(float)/pricing_data['T5YIFR']
    pricing_data['T5YIFR_m_spread'] = pricing_data['T5YIFR'].rolling(30).mean().shift().astype(float)/pricing_data['T5YIFR']
    pricing_data['T5YIFR_6monthly_return'] = pricing_data['T5YIFR'].pct_change(periods  = 176) .astype(float)
    pricing_data['T5YIFR_12monthly_return'] = pricing_data['T5YIFR'].pct_change(periods  = 252) .astype(float)
    pricing_data['T5YIFR_36monthly_return'] = pricing_data['T5YIFR'].pct_change(periods  = 500) .astype(float)
    
    pricing_data['BAMLH0A0HYM2_3y_spread'] = pricing_data['BAMLH0A0HYM2'].rolling(750).mean().shift().astype(float)/pricing_data['BAMLH0A0HYM2']
    pricing_data['BAMLH0A0HYM2_y_spread'] = pricing_data['BAMLH0A0HYM2'].rolling(250).mean().shift().astype(float)/pricing_data['BAMLH0A0HYM2']
    pricing_data['BAMLH0A0HYM2_sa_spread'] = pricing_data['BAMLH0A0HYM2'].rolling(90).mean().shift().astype(float)/pricing_data['BAMLH0A0HYM2']
    pricing_data['BAMLH0A0HYM2_m_spread'] = pricing_data['BAMLH0A0HYM2'].rolling(30).mean().shift().astype(float)/pricing_data['BAMLH0A0HYM2']
    pricing_data['BAMLH0A0HYM2_6monthly_return'] = pricing_data['BAMLH0A0HYM2'].pct_change(periods  = 176) .astype(float)
    pricing_data['BAMLH0A0HYM2_12monthly_return'] = pricing_data['BAMLH0A0HYM2'].pct_change(periods  = 252) .astype(float)
    pricing_data['BAMLH0A0HYM2_36monthly_return'] = pricing_data['BAMLH0A0HYM2'].pct_change(periods  = 750) .astype(float)
    
    
    final = pricing_data.merge(Balance_sheet1,how = 'outer', left_on = ['Date'], right_on = ['Date'] )
    final = final.merge(earnings_from_Zach,how = 'outer', left_on = ['Date'], right_on = ['Date'] )
    final = final.dropna(subset=['stock'])
    del final['Ticker'], final['DATE']
    final = final.sort_values('Date', ascending = True).fillna(method = 'ffill')
    
    
    final['MarketCap'] = final['sharesOutstanding']*final['Adj Close']
    
    try:
        final['Assets%/Price'] = final['Assets']/final['Adj Close']
    except:
        final['Assets%/Price'] = np.nan     
    try:  
        final['Liabilities%/Price'] = final['Liabilities']/final['Adj Close']
    except:
        final['Liabilities%/Price'] = np.nan
    try:  
        final['Revenue%/Price'] = final[' Actual Revenue']/final['sharesOutstanding']/final['Adj Close']
    except:
        final['Revenue%/Price'] = np.nan
    try:  
        final['CurrentAssets%/Price'] = final['Assets, Current']/final['Adj Close']
    except:
        final['CurrentAssets%/Price'] = np.nan
    try:  
        final['CurrentLiabilities%/Price'] = final['Liabilities, Current']/final['Adj Close']
    except:
        final['CurrentLiabilities%/Price'] = np.nan
    try:  
        final['Equity%/Price'] = final['Equity/Share']/final['Adj Close']
    except:
        final['Equity%/Price'] = np.nan
    try:
        final['AccountsReceive/Price'] = final['Accounts Receivable']/final['Adj Close']
    except:
        final['AccountsReceive/Price'] = np.nan
    try:
        final['AccountsPay/Price'] = final['Accounts Payable, Current']/final['Adj Close']
    except:
        final['AccountsReceive/Price'] = np.nan       
    try:
        final['R&D/Price'] = final['R&D']/final['Adj Close']
    except:
        final['R&D/Price'] = np.nan 
    try:
        final['Inventory/Price'] = final['Inventory, Net']/final['Adj Close']
    except:
        final['Inventory/Price']  = np.nan
    try:
        final['Operating Income/Price'] = final['Operating Income (Loss)']/final['Adj Close']
    except:
        final['Inventory/Price']  = np.nan
    
    final['AnnualPE'] = final['Adj Close'] / (final[' Reported EPS'].rolling(250).mean().shift().astype(float)*4)
    final['AnnualPE_average'] = final[' Reported EPS'].mean()*4/ final['AnnualPE']
    
    final['AnnualPE_y'] = (final['AnnualPE'].rolling(250).mean().shift().astype(float)*4)/ final['AnnualPE']
    final['AnnualPE_1day'] = final['AnnualPE'].pct_change(periods  = 1) .astype(float)
    final['AnnualPE_1week_return'] = final['AnnualPE'].pct_change(periods  = 5) .astype(float)
    final['AnnualPE_1monthly_return'] = final['AnnualPE'].pct_change(periods  = 22) .astype(float)
    final['AnnualPE_6monthly_return'] = final['AnnualPE'].pct_change(periods  = 176) .astype(float)
    final['AnnualPE_12monthly_return'] = final['AnnualPE'].pct_change(periods  = 252) .astype(float)
    
    try:
        final['R&D_y_spread'] = final['R&D'].rolling(250).mean().shift().astype(float)/final['R&D']
        final['R&D_week_return'] = final['R&D'].pct_change(periods  = 5) .astype(float)
        final['R&D_monthly_return'] = final['R&D'].pct_change(periods  = 22) .astype(float)
        final['R&D_6monthly_return'] = final['R&D'].pct_change(periods  = 176) .astype(float)
        final['R&D_12monthly_return'] = final['R&D'].pct_change(periods  = 252) .astype(float)
    except:
        final['R&D_y_spread'] = np.nan
        final['R&D_week_return'] = np.nan
        final['R&D_monthly_return'] = np.nan
        final['R&D_6monthly_return'] = np.nan
        final['R&D_12monthly_return'] = np.nan
    try:   
        final['Assets_y_spread'] = final['Assets'].rolling(250).mean().shift().astype(float)/final['Assets']
        final['Assets_6monthly_return'] = final['Assets'].pct_change(periods  = 176) .astype(float)
        final['Assets_12monthly_return'] = final['Assets'].pct_change(periods  = 252) .astype(float)
        final['Assets_a_spread'] = final['Assets'].rolling(176).mean().shift().astype(float)/final['Assets']
    except:
        print ('Assets missing')  
    try: 
        final['Liabilities_y_spread'] = final['Liabilities%/Price'].rolling(250).mean().shift().astype(float)/final['Liabilities%/Price']
        final['Liabilities_6monthly_return'] = final['Liabilities'].pct_change(periods  = 176) .astype(float)
        final['Liabilities_12monthly_return'] = final['Liabilities'].pct_change(periods  = 252) .astype(float)
        final['Liabilities_a_spread'] = final['Liabilities%/Price'].rolling(176).mean().shift().astype(float)/final['Liabilities%/Price']
    except:
        print ('Liabilitie missing')   
   
    try:  
        final['Cash/Share_y_spread'] = final['Cash/Share'].rolling(250).mean().shift().astype(float)/final['Cash/Share']
        final['Cash/Share_6monthly_return'] = final['Cash/Share'].pct_change(periods  = 176) .astype(float)
        final['Cash/Share_12monthly_return'] = final['Cash/Share'].pct_change(periods  = 252) .astype(float)
        final['Cash/Share_a_spread'] = final['Cash/Share'].pct_change(periods  = 176) .astype(float)
    except:
        print ('cash missing')
        
    final['count'] = final.groupby('Assets').cumcount()/100
    final = final.dropna(axis=1, how='all')
    final = final[final['Assets'].notna()]
    final = final.replace([np.inf, -np.inf], np.nan)
    final = final.fillna(method='ffill')
    final = final.fillna(method='bfill')
    
    try:
        final[' Revenue Estimate'] = final['O Revenue Estimate']/final['sharesOutstanding']
        final[' Actual Revenue'] = final[' Actual Revenue']/final['sharesOutstanding']
    except:
        final['Inventory/Price']  = np.nan
    del final['MarketCap'], final['sharesOutstanding']
    
    final.to_csv('final_data\\'+ stock + str(days_to_pred) +'.csv', index=False)
    return final

def get_balace_sheet(stock):
    print (stock)
    cik = pd.read_csv('sec_files\\cik.csv')
    cik['cik'] = cik['cik'].astype(str).str.zfill(10)
    cikI = cik[cik['Ticker'] == stock.lower()]
    cikI = cikI.values.tolist()[0][1]
    pili = yf.Ticker(stock)
    a = pili.info['sharesOutstanding']
    with open("company_facts\\CIK" + cikI +".json") as f:
        data = json.load(f)
    yolo1 = data['facts']
    yolo1 = yolo1['us-gaap']
    new_df = return_data_from_EDGAR('Assets', stock)
    new_df['Assets'] = a
    index = new_df.index
    new_df = new_df.rename(columns={'Assets': 'sharesOutstanding'})
    for key in yolo1.keys():
        i = return_data_from_EDGAR(key, stock)
        if i is None:
            new_df[key] = np.nan
        else:
            index_i = i.index
            if len(index_i) > (len(index) - 8):
                try:
                    new_df = pd.merge(new_df, i,  how='left', left_on=['Ticker','Date'], right_on = ['Ticker','Date'])
                except:
                    new_df[key] = np.nan
                    continue                
    try:
        new_df['Liabilities'] = new_df.Liabilities.fillna(new_df['Liabilities and Equity']-new_df["Stockholders' Equity Attributable to Parent"])
        new_df["Stockholders' Equity Attributable to Parent"] = new_df["Stockholders' Equity Attributable to Parent"].fillna(new_df['Liabilities and Equity']-new_df["Liabilities"])
        new_df["Stockholders' Equity Attributable to Parent"] = new_df["Stockholders' Equity Attributable to Parent"].fillna(new_df['Assets']-new_df["Liabilities"])
        new_df['Liabilities'] = new_df.Liabilities.fillna(new_df['Assets']-new_df["Stockholders' Equity Attributable to Parent"])
        new_df["Stockholders' Equity Attributable to Parent"] = new_df["Stockholders' Equity Attributable to Parent"].fillna(new_df['Liabilities and Equity']-new_df["Liabilities"])
        new_df["Stockholders' Equity Attributable to Parent"] = new_df["Stockholders' Equity Attributable to Parent"].fillna(new_df['Assets']-new_df["Liabilities"])
        A = new_df['Ticker'].copy()
        B = new_df['Date'].copy()
        new_df = pd.concat([new_df.ffill(), new_df.bfill()]).groupby(level=0).mean()
        new_df = new_df.divide(new_df["sharesOutstanding"], axis="index")
        new_df.insert(0, 'Date', B)
        new_df.insert(0, 'Ticker', A )
    except:
        print ('gamithike ligo')
    try:
        new_df['ASSET/DEBT'] = new_df['Assets']/new_df['Liabilities']
    except:
        new_df['ASSET/DEBT'] = np.nan
    try:
        new_df['P/B'] = (new_df['Assets'] - new_df['Liabilities'])
    except:
        new_df['P/B'] = np.nan
    try:
        new_df['current_ratio'] = new_df['Assets, Current']/new_df['Liabilities, Current']
    except:
        new_df['current_ratio'] = np.nan
    try:
        new_df = new_df.rename(columns={"Stockholders' Equity Attributable to Parent": "Equity/Share", "Net Income (Loss) Attributable to Parent": "EPS"})
        new_df = new_df.rename(columns={"Cash and Cash Equivalents, at Carrying Value": "Cash/Share", "Research and Development Expense": "R&D"})
        new_df = new_df.rename(columns={"Accounts Receivable, after Allowance for Credit Loss, Current": "Accounts Receivable", "ResearchAndDevelopmentExpense": "R&D"}) 
    except:
        print('gamithike kai allo')
    
    try:
        new_df = new_df.rename(columns={"Stockholders' Equity Attributable to Parent": "Equity/Share", "Net Income (Loss) Attributable to Parent": "EPS"})
        new_df = new_df.rename(columns={"Cash and Cash Equivalents, at Carrying Value": "Cash/Share", "Research and Development Expense": "R&D"})
        new_df = new_df.rename(columns={"Accounts Receivable, after Allowance for Credit Loss, Current": "Accounts Receivable", "ResearchAndDevelopmentExpense": "R&D"}) 
    except:
        print('gamithike kai allo')
    
    new_df['sharesOutstanding'] = a
    new_df = new_df.dropna(axis=1, how='all')
    new_df.to_csv('Balance_sheet\\'+stock+'.csv', index=False) 
    print (stock + ' Done')
    return new_df

def get_fact_per_sec(stock):
    cik = pd.read_csv('sec_files\\cik.csv')
    cik['cik'] = cik['cik'].astype(str).str.zfill(10)
    cikI = cik[cik['Ticker'] == stock.lower()]
    cikI = cikI.values.tolist()[0][1]
    link = "https://data.sec.gov/api/xbrl/companyfacts/CIK" + cikI +".json"
    hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}
    req = requests.get(link,headers=hdr)
    content = req.content
    data = json.loads(content)
    with open("company_facts\\CIK" + cikI +".json", "w+") as f:
        json.dump(data, f)
    return data

def update_CIK_file():
    #save_path = 'sec_files\\'
    link = 'https://www.sec.gov/include/ticker.txt'
    hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}
    response = requests.get(link,headers=hdr)
    with open("cik.txt", "w") as f:
        f.write(response.text)
    read_file = pd.read_csv (r'cik.txt',delimiter=r"\s+", names = ['Ticker','cik'] )
    read_file.to_csv (r'sec_files\cik.csv', index=None)
    file_path = 'cik.txt'
    os.remove(file_path)

def download_EDGAR_data():
    hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}
    r = requests.get('https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip',headers=hdr)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("/company_facts")

def return_data_from_EDGAR(metric, stock):
    cik = pd.read_csv('sec_files\\cik.csv')
    cik['cik'] = cik['cik'].astype(str).str.zfill(10)
    cikI = cik[cik['Ticker'] == stock.lower()]
    cikI = cikI.values.tolist()[0][1]
    with open("company_facts\\CIK" + cikI +".json") as f:
        data = json.load(f)
        # Print the type of data variable

    try:
        date_list = []
        value_date = []
        ticker_date = []
        yolo1 = data['facts']
        yolo1 = yolo1['us-gaap']
        yolo1 = yolo1[metric]
        label = yolo1['label']
        yolo1 = yolo1['units']
        yolo1 = yolo1['USD']
        for y in yolo1:     
            if y['form'] == '20-F':
                try:
                    check1 = y['filed']
                    check2 = y['end']
                    d1 = datetime.strptime(check1, '%Y-%m-%d')
                    d2 = datetime.strptime(check2, '%Y-%m-%d')
                    diff = (d1 - d2).days
                    ticker_date.append(stock)
                    date_list.append(y['filed'])
                    value_date.append(y['val'])
                except:
                    print ('debug')
            else:
                try:
                    check1 = y['filed']
                    check2 = y['end']
                    check3 = y['start']
                    d1 = datetime.strptime(check1, '%Y-%m-%d')
                    d2 = datetime.strptime(check2, '%Y-%m-%d')
                    d3 = datetime.strptime(check3, '%Y-%m-%d')
                    diff = (d1 - d2).days
                    diff2 = (d2 - d3).days
                    if diff < 45 and diff2 < 100:
                        ticker_date.append(stock)
                        date_list.append(y['filed'])
                        value_date.append(y['val'])
                except:
                    try:
                        check1 = y['filed']
                        check2 = y['end']
                        d1 = datetime.strptime(check1, '%Y-%m-%d')
                        d2 = datetime.strptime(check2, '%Y-%m-%d')
                        diff = (d1 - d2).days
                        if diff < 90:
                            ticker_date.append(stock)
                            date_list.append(y['filed'])
                            value_date.append(y['val'])
                    except: 
                        continue
        df1 = pd.DataFrame({'Ticker': ticker_date, 'Date': date_list,label: value_date})
        df1 = df1.groupby(['Ticker','Date'], sort=False)[label].max()
        df1 = df1.reset_index()
        return df1
    except:
        pass


def EntityCommonStockSharesOutstanding(stock):
    cik = pd.read_csv('sec_files\\cik.csv')
    cik['cik'] = cik['cik'].astype(str).str.zfill(10)
    cikI = cik[cik['Ticker'] == stock]
    cikI = cikI.values.tolist()[0][1]
    pili = yf.Ticker(stock)
    a = pili.info['sharesOutstanding']
    with open("company_facts\\CIK" + cikI +".json") as f:
        data = json.load(f)
    try:
        date_list = []
        value_date = []
        ticker_date = []
        yolo1 = data['facts']
        yolo1 = yolo1['us-gaap']
        yolo1 = yolo1['Assets']
        yolo1 = yolo1['units']
        yolo1 = yolo1['USD']
        for y in yolo1:
            ticker_date.append(stock)
            date_list.append(y['filed'])
            value_date.append(a)
        percentile_list1 = pd.DataFrame({'Ticker': ticker_date, 'Date': date_list,'sharesOutstanding': value_date})
        percentile_list1 = percentile_list1.drop_duplicates()
        return percentile_list1
    except:
        pass

def flatten(t):
    return [item for sublist in t for item in sublist]


def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)


def get_earnings_from_Zach(ticker):
    df2 = pd.read_csv('sec_files\\cik.csv') 
    df2 = df2[(df2['Ticker'] == ticker.lower())]
    market = df2.values.tolist()[0][2]
    url = requests.get('https://www.marketbeat.com/stocks/' + 'NASDAQ' + '/'+ ticker +'/earnings/')
    x = str(url.text)
    x = str(html2text.html2text(x))
    
    indexe = 'Earnings History by Quarter'
    head, sep, tail = x.partition(indexe)
    y = str(tail)
    yolo = 'Earnings Frequently Asked Questions'
    head, sep, tail = y.partition(yolo)
    
    head = '\t'.join([line.strip() for line in head.split('\n')])
    p = head
    p = p.replace("($", "-")
    p = p.replace(")| ", " | ")
    
    modified_string = re.sub(r"\([^()]*\)", "", p)
    
    
    
    modified_string = modified_string.replace("	", "")
    modified_string = modified_string.replace(" billion", "0000000")
    modified_string = modified_string.replace("billion", "0000000")
    modified_string = modified_string.replace(" million", "0000")
    modified_string = modified_string.replace("million", "0000")
    modified_string = remove_text_inside_brackets(modified_string,"[]")
    modified_string = modified_string.replace("!", "")
    modified_string = modified_string.replace("_", "")
    modified_string = modified_string.replace("#", "")
    modified_string = modified_string.replace("$", "")
    modified_string = modified_string.replace("|", ",")
    yolo = modified_string.split(",")
    
    final = []
    grgre = True
    while grgre == True:  
        tretre = yolo[0:8]
        final.append(tretre)
        del yolo[0:8]
        if len(yolo) < 8:
            grgre = False
            
            
    df = pd.DataFrame(final)
    df = df.rename(columns=df.iloc[0])
    df = df.iloc[1: , :]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df[' RevenueEstimate'] = df[' RevenueEstimate'].str.replace('\.','')
    df[' RevenueEstimate'] = df[' RevenueEstimate'].str.replace(' ','')
    df[' Actual Revenue'] = df[' Actual Revenue'].str.replace("\.", "")
    df[' Actual Revenue'] = df[' Actual Revenue'].str.replace(" ", "")
    df = df.drop(columns=[' Quarter'])
    
    fix = [' Consensus Estimate', ' Reported EPS', ' Beat/Miss', ' GAAP EPS',' RevenueEstimate', ' Actual Revenue']
    for i in fix:
        #df[i] = [p.sub('', x) for x in df[i]]
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df[' Revenue Surprise'] = df[' Actual Revenue'] / df[' RevenueEstimate'] - 1
    df[' Reported EPS'] = df[' Reported EPS'].fillna(0)
    df[' Consensus Estimate'] = df[' Consensus Estimate'].fillna(0)
    df[' Beat/Miss'] = df[' Beat/Miss'].fillna(0)
    df[' GAAP EPS'] = df[' GAAP EPS'].fillna(0)
    df.to_csv('Earnings\\'+ticker+'_earnings.csv', index=False)
    return df

def final_dataset(stock, days_to_predict_in_the_future):
    days_in_past = 10000
    startdate = datetime.today() - timedelta(days=days_in_past)
    startdate = startdate.strftime("%Y-%m-%d")
    enddate = date.today()
    Balance_sheet1 = pd.read_csv('portfolios\\'+ stock +'.csv') 
    Balance_sheet1['Date'] = pd.to_datetime(Balance_sheet1['Date'], errors='coerce')
    earnings_from_Zach = get_earnings_from_Zach(stock)
    earnings_from_Zach = pd.read_csv('Earnings\\'+ stock +'_earnings.csv')
    earnings_from_Zach['Date'] = pd.to_datetime(earnings_from_Zach['Date'], errors='coerce')
    fed_data = get_fed_date()
    pricing_data = create_weekly_candle_dataset([stock],days_to_predict_in_the_future,startdate,enddate)
    pricing_data = pricing_data.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )
    final = pricing_data.merge(Balance_sheet1,how = 'outer', left_on = ['Date'], right_on = ['Date'] )
    final = final.merge(earnings_from_Zach,how = 'outer', left_on = ['Date'], right_on = ['Date'] )
    final = final.sort_values('Date', ascending = True).fillna(method = 'ffill')
    final = final.dropna(subset=['stock'])
    del final['Ticker'], final['DATE']
    return final


def get_stocks_by_nlp(my_string):
  data = pd.read_csv('company_profile.csv')
  for word in my_string.split():
      docs = data['profile'].tolist()
      vectorizer = TfidfVectorizer()
      X = vectorizer.fit_transform(docs)
      df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())
      q = [my_string]
      q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
      sim = {}
      for i in range(7041):
          sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
      sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
      df2 = pd.DataFrame( columns = ['Ticker' ,'profile'])
      for k, v in sim_sorted:
          if v > 0.05:
            df2 = df2.append(data[data['profile'] == docs[k]])
      df2.reset_index(level=0, inplace=True)
      list_of_stocks = df2['Ticker'].to_list()
  return list_of_stocks



def get_fed_date():
    days_in_past = 20000
    startdate = datetime.today() - timedelta(days=days_in_past)
    startdate = startdate.strftime("%Y-%m-%d")
    enddate =  datetime.today() 
    enddate =  enddate.strftime("%Y-%m-%d")
    list1 = ['T5YIFR',	'BAMLH0A0HYM2',	'SAHMCURRENT',	'WLEMUINDXD',	'T10Y2Y',	'TOTALSA',	
             'PCU327320327320',	'RRPONTSYD',	'FEDFUNDS',	'MORTGAGE30US',	'WALCL',	
             'BAMLH0A3HYC',	'DEXCHUS',	'DHHNGSP',	'DEXUSUK',	'NIKKEI225',	'DJFUELUSGULF',	
             'MEHOINUSA672N',	'ICSA',	'FYFSD',	'JTSJOL',	'EMRATIO',	'PNGASEUUSDM',	
             'FRGSHPUSM649NCIS',	'PIORECRUSDM',	'DTWEXAFEGS',	
             'PALUMUSDM',	'PSOYBUSDQ',	'VXFXICLS']
    fed_data = web.DataReader(list1, 'fred', startdate, enddate)
    fed_data.iloc[:, 1] = pd.to_numeric(fed_data.iloc[:, 1],errors = 'coerce')
    fed_data = fed_data.fillna(method = 'ffill')
    fed_data.reset_index(inplace=True)
    corr_matrix = fed_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df = fed_data.drop(fed_data[to_drop], axis=1)    
    lista = df.columns.tolist()
    for i in lista:
        if i not in ['DATE']:
            df[i + '/EMA_5'] = df[i]/df[i].rolling(5).mean().shift().astype(float)
            df[i + '.EMA_5/EMA_15'] = df[i].rolling(5).mean().shift().astype(float)/df[i].rolling(15).mean().shift().astype(float)
            df[i + '.SMA_50/SMA_200'] = df[i].rolling(50).mean().shift().astype(float)/df[i].rolling(200).mean().shift().astype(float)
            df[i + '.SMA_200/SMA_1000'] = df[i].rolling(200).mean().shift().astype(float)/df[i].rolling(1000).mean().shift().astype(float)
            df[i + '5day_return'] = df[i].pct_change(periods  = 5) .astype(float)
            df[i + '10day_return'] = df[i].pct_change(periods  = 10) .astype(float)
            df[i + '50day_return'] = df[i].pct_change(periods  = 50) .astype(float)
            df[i + '100day_return'] = df[i].pct_change(periods  = 50) .astype(float)
            df[i + '100day_return'] = df[i].pct_change(periods  = 200) .astype(float)
            df[i + '100day_return'] = df[i].pct_change(periods  = 1000) .astype(float)
            df[i + '200day_return'] = df[i].rolling(200).mean().shift().pct_change(periods  = 10) .astype(float)
            df[i + '.2nd_derivative1'] = df[i + '.EMA_5/EMA_15']/ df[i + '/EMA_5']
            df[i + '.2nd_derivative2'] = df[i + '.SMA_50/SMA_200'] / df[i + '.EMA_5/EMA_15'] 
            df[i + '.2nd_derivative3'] = df[i + '.SMA_200/SMA_1000'] / df[i + '.SMA_50/SMA_200']
        else:
            continue
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    df = df.drop(df[to_drop], axis=1)   
    return df

def RSI(df,n):
    close = df['Adj Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def getPortfolioVol(weights, meanReturns, covMatrix):
    return calcPortfolioPerformance(weights, meanReturns, covMatrix)[1]

def candle_dataset(stock,future_period_tbpredict,start,end):   
    try:
        df = yf.download(stock)
        df = df.loc[start:end]
        df = df.copy()
        dfa = pd.read_csv('sec_files\\stockid.csv')
        dfa = dfa.loc[dfa['stock'] == stock]
        subindustryid = dfa['subsectorid'].item()
        df['Date'] = df.index
        df['weekday'] = df['Date'].dt.dayofweek
        del df['Date'] 
        df.insert(0, 'stock', stock)
        df.insert(1, 'subsector',subindustryid)
        df['Volume'] = df['Volume'].replace(0, 1)
        del df['Close']
        df['Volume_Rate'] = df['Volume'] / df['Volume'].shift() -1
        df['HLspread'] = (df['High'] / df['Low'])-1
        df['COC_spread'] = (df['Adj Close'] / df['Open']) -1
        df['CLC_spread'] = (df['Adj Close'] / df['Low'])-1
        df['Gap'] = df['Open']/ df['Adj Close'].shift() - 1  
        
        df['1Volume_Rate'] = (df['Volume'].shift(periods = 1) / df['Volume'].shift(periods = 2)) -1
        df['1HLspread'] = (df['High'].shift(periods = 1) / df['Low']).shift(periods = 1) -1
        df['1COC_spread'] = (df['Adj Close'].shift(periods = 1) / df['Open'].shift(periods = 1)) -1
        df['1CLC_spread'] = (df['Adj Close'].shift(periods = 1)/ df['Low']) - 1
        df['1Gap'] = (df['Open'].shift(periods = 1)/ df['Adj Close'].shift(periods = 2)) - 1   
        
        df['2Volume_Rate'] = (df['Volume'].shift(periods = 2) / df['Volume'].shift(periods = 3)) -1
        df['2HLspread'] = (df['High'].shift(periods = 2) / df['Low']).shift(periods = 2) -1
        df['2COC_spread'] = (df['Adj Close'].shift(periods = 2) / df['Open'].shift(periods = 2)) -1
        df['2CLC_spread'] = (df['Adj Close'].shift(periods = 2)/ df['Low']) - 1
        df['2Gap'] = (df['Open'].shift(periods = 2)/ df['Adj Close'].shift(periods = 3)) - 1  
        
        df['3Volume_Rate'] = (df['Volume'].shift(periods = 3) / df['Volume'].shift(periods = 4)) -1
        df['3HLspread'] = (df['High'].shift(periods = 3) / df['Low']).shift(periods = 3) -1
        df['3COC_spread'] = (df['Adj Close'].shift(periods = 3) / df['Open'].shift(periods = 3)) -1
        df['3CLC_spread'] = (df['Adj Close'].shift(periods = 3)/ df['Low']) - 1
        df['3Gap'] = (df['Open'].shift(periods = 3)/ df['Adj Close'].shift(periods = 4)) - 1  
        
        df['4Volume_Rate'] = (df['Volume'].shift(periods = 4) / df['Volume'].shift(periods = 5)) -1
        df['4HLspread'] = (df['High'].shift(periods = 4) / df['Low']).shift(periods = 4) -1
        df['4COC_spread'] = (df['Adj Close'].shift(periods = 4) / df['Open'].shift(periods = 4)) -1
        df['4CLC_spread'] = (df['Adj Close'].shift(periods = 4)/ df['Low']) - 1
        df['4Gap'] = (df['Open'].shift(periods = 4)/ df['Adj Close'].shift(periods = 5)) - 1
        
        
        ###############################################################################
        
        df['5Volume_Rate'] = (df['Volume'].shift(periods = 5) / df['Volume'].shift(periods = 6)) -1
        df['5HLspread'] = (df['High'].shift(periods = 5) / df['Low']).shift(periods = 5) -1
        df['5COC_spread'] = (df['Adj Close'].shift(periods = 5) / df['Open'].shift(periods = 6)) -1
        df['5CLC_spread'] = (df['Adj Close'].shift(periods = 5)/ df['Low']) - 1
        df['5Gap'] = (df['Open'].shift(periods = 5)/ df['Adj Close'].shift(periods = 5)) - 1
        
        df['6Volume_Rate'] = (df['Volume'].shift(periods = 6) / df['Volume'].shift(periods = 7)) -1
        df['6HLspread'] = (df['High'].shift(periods = 6) / df['Low']).shift(periods = 6) -1
        df['6COC_spread'] = (df['Adj Close'].shift(periods = 6) / df['Open'].shift(periods = 7)) -1
        df['6CLC_spread'] = (df['Adj Close'].shift(periods = 6)/ df['Low']) - 1
        df['6Gap'] = (df['Open'].shift(periods = 6)/ df['Adj Close'].shift(periods = 6)) - 1
        
        df['7Volume_Rate'] = (df['Volume'].shift(periods = 7) / df['Volume'].shift(periods = 8)) -1
        df['7HLspread'] = (df['High'].shift(periods = 7) / df['Low']).shift(periods = 7) -1
        df['7COC_spread'] = (df['Adj Close'].shift(periods = 7) / df['Open'].shift(periods = 8)) -1
        df['7CLC_spread'] = (df['Adj Close'].shift(periods = 7)/ df['Low']) - 1
        df['7Gap'] = (df['Open'].shift(periods = 7)/ df['Adj Close'].shift(periods = 7)) - 1
        
        df['8Volume_Rate'] = (df['Volume'].shift(periods = 8) / df['Volume'].shift(periods = 9)) -1
        df['8HLspread'] = (df['High'].shift(periods = 8) / df['Low']).shift(periods = 8) -1
        df['8COC_spread'] = (df['Adj Close'].shift(periods = 8) / df['Open'].shift(periods = 9)) -1
        df['8CLC_spread'] = (df['Adj Close'].shift(periods = 8)/ df['Low']) - 1
        df['8Gap'] = (df['Open'].shift(periods = 8)/ df['Adj Close'].shift(periods = 8)) - 1
        
        df['9Volume_Rate'] = (df['Volume'].shift(periods = 9) / df['Volume'].shift(periods = 10)) -1
        df['9HLspread'] = (df['High'].shift(periods = 9) / df['Low']).shift(periods = 9) -1
        df['9COC_spread'] = (df['Adj Close'].shift(periods = 9) / df['Open'].shift(periods = 10)) -1
        df['9CLC_spread'] = (df['Adj Close'].shift(periods = 9)/ df['Low']) - 1
        df['9Gap'] = (df['Open'].shift(periods = 9)/ df['Adj Close'].shift(periods = 9)) - 1
        
        df['10Volume_Rate'] = (df['Volume'].shift(periods = 10) / df['Volume'].shift(periods = 11)) -1
        df['10HLspread'] = (df['High'].shift(periods = 10) / df['Low']).shift(periods = 10) -1
        df['10COC_spread'] = (df['Adj Close'].shift(periods = 10) / df['Open'].shift(periods = 11)) -1
        df['10CLC_spread'] = (df['Adj Close'].shift(periods = 10)/ df['Low']) - 1
        df['10Gap'] = (df['Open'].shift(periods = 10)/ df['Adj Close'].shift(periods = 10)) - 1
        
        df['11Volume_Rate'] = (df['Volume'].shift(periods = 11) / df['Volume'].shift(periods = 12)) -1
        df['11HLspread'] = (df['High'].shift(periods = 11) / df['Low']).shift(periods = 11) -1
        df['11COC_spread'] = (df['Adj Close'].shift(periods = 11) / df['Open'].shift(periods = 12)) -1
        df['11CLC_spread'] = (df['Adj Close'].shift(periods = 11)/ df['Low']) - 1
        df['11Gap'] = (df['Open'].shift(periods = 11)/ df['Adj Close'].shift(periods = 11)) - 1
        
        df['12Volume_Rate'] = (df['Volume'].shift(periods = 12) / df['Volume'].shift(periods = 13)) -1
        df['12HLspread'] = (df['High'].shift(periods = 12) / df['Low']).shift(periods = 12) -1
        df['12COC_spread'] = (df['Adj Close'].shift(periods = 12) / df['Open'].shift(periods = 13)) -1
        df['12CLC_spread'] = (df['Adj Close'].shift(periods = 12)/ df['Low']) - 1
        df['12Gap'] = (df['Open'].shift(periods = 12)/ df['Adj Close'].shift(periods = 12)) - 1
        
        df['13Volume_Rate'] = (df['Volume'].shift(periods = 13) / df['Volume'].shift(periods = 14)) -1
        df['13HLspread'] = (df['High'].shift(periods = 13) / df['Low']).shift(periods = 13) -1
        df['13COC_spread'] = (df['Adj Close'].shift(periods = 13) / df['Open'].shift(periods = 14)) -1
        df['13CLC_spread'] = (df['Adj Close'].shift(periods = 13)/ df['Low']) - 1
        df['13Gap'] = (df['Open'].shift(periods = 13)/ df['Adj Close'].shift(periods = 13)) - 1
        
        df['14Volume_Rate'] = (df['Volume'].shift(periods = 14) / df['Volume'].shift(periods = 15)) -1
        df['14HLspread'] = (df['High'].shift(periods = 14) / df['Low']).shift(periods = 14) -1
        df['14COC_spread'] = (df['Adj Close'].shift(periods = 14) / df['Open'].shift(periods = 15)) -1
        df['14CLC_spread'] = (df['Adj Close'].shift(periods = 14)/ df['Low']) - 1
        df['14Gap'] = (df['Open'].shift(periods = 14)/ df['Adj Close'].shift(periods = 14)) - 1
        
        df['15Volume_Rate'] = (df['Volume'].shift(periods = 15) / df['Volume'].shift(periods = 16)) -1
        df['15HLspread'] = (df['High'].shift(periods = 15) / df['Low']).shift(periods = 15) -1
        df['15COC_spread'] = (df['Adj Close'].shift(periods = 15) / df['Open'].shift(periods = 16)) -1
        df['15CLC_spread'] = (df['Adj Close'].shift(periods = 15)/ df['Low']) - 1
        df['15Gap'] = (df['Open'].shift(periods = 15)/ df['Adj Close'].shift(periods = 15)) - 1
        
        df['16Volume_Rate'] = (df['Volume'].shift(periods = 16) / df['Volume'].shift(periods = 17)) -1
        df['16HLspread'] = (df['High'].shift(periods = 16) / df['Low']).shift(periods = 16) -1
        df['16COC_spread'] = (df['Adj Close'].shift(periods = 16) / df['Open'].shift(periods = 17)) -1
        df['16CLC_spread'] = (df['Adj Close'].shift(periods = 16)/ df['Low']) - 1
        df['16Gap'] = (df['Open'].shift(periods = 16)/ df['Adj Close'].shift(periods = 16)) - 1

        ################################################################################

        df['EMA_5'] = df['Adj Close'].rolling(5).mean().shift().astype(float)
        df['EMA_15'] = df['Adj Close'].rolling(15).mean().shift().astype(float)
        df['SMA_50'] = df['Adj Close'].rolling(50).mean().shift().astype(float)
        df['SMA_100'] = df['Adj Close'].rolling(100).mean().shift().astype(float)
        df['SMA_200'] = df['Adj Close'].rolling(200).mean().shift() .astype(float)
        df['EMA_5/EMA_15'] = df['EMA_5']/df['EMA_15']
        df['EMA_5/SMA_100'] = df['EMA_5']/df['SMA_100']
        df['EMA_5/SMA_50'] = df['EMA_5']/df['SMA_50']
        df['EMA_5/SMA_200'] = df['EMA_5']/df['SMA_200']        
        df['EMA_15/SMA_200'] = df['EMA_15']/df['SMA_200']
        df['SMA_50/SMA_200'] = df['SMA_50']/df['SMA_200']
        df['SMA_100/SMA_200'] = df['SMA_100']/df['SMA_200']
        df['EMA_15/SMA_100'] = df['EMA_15']/df['SMA_100']
        df['EMA_15/SMA_50'] = df['EMA_15']/df['SMA_50']
        df['RSI_5'] = RSI(df, n=5).astype(float)/100
        df['RSI_14'] = RSI(df, n=14).astype(float)/100
        df['RSI_30'] = RSI(df, n=30).astype(float)/100
        df['RSI_50'] = RSI(df, n=50).astype(float)/100
        df['10day_return'] = df['Adj Close'].pct_change(periods  = 10) .astype(float)
        df['monthly_return'] = df['Adj Close'].pct_change(periods  = 22) .astype(float)
        df['2monthly_return'] = df['Adj Close'].pct_change(periods  = 44) .astype(float)
        df['3monthly_return'] = df['Adj Close'].pct_change(periods  = 66) .astype(float)
        df['6monthly_return'] = df['Adj Close'].pct_change(periods  = 132) .astype(float)
        df['12monthly_return'] = df['Adj Close'].pct_change(periods  = 252) .astype(float)
        df['24monthly_return'] = df['Adj Close'].pct_change(periods  = 320) .astype(float)
        df['36monthly_return'] = df['Adj Close'].pct_change(periods  = 470) .astype(float)
        df['60monthly_return'] = df['Adj Close'].pct_change(periods  = 900) .astype(float)
        df['monthly_volume_pct_change'] = df['Volume'].pct_change(periods  = 22) .astype(float)
        df['3months_volume_pct_change'] = df['Volume'].pct_change(periods  = 66) .astype(float)
        df['yearly_volume_pct_change'] = df['Volume'].pct_change(periods  = 252) .astype(float)
        df['24month_volume_pct_change'] = df['Volume'].pct_change(periods  = 320) .astype(float)
        df['36month_volume_pct_change'] = df['Volume'].pct_change(periods  = 470) .astype(float)
        df['60month_volume_pct_change'] = df['Volume'].pct_change(periods  = 900) .astype(float)
                
        df['std90'] = df['Adj Close'].rolling(66).std()
        df['std180'] = df['Adj Close'].rolling(126).std()
        df['std1y'] = df['Adj Close'].rolling(252).std()
        df['std3y'] = df['Adj Close'].rolling(750).std()
        
        df['min90'] = df['Adj Close']/ df['Adj Close'].rolling(66).min()
        df['min180'] = df['Adj Close']/ df['Adj Close'].rolling(126).min()
        df['min1y'] = df['Adj Close'] / df['Adj Close'].rolling(252).min()
        df['min3y'] = df['Adj Close'] / df['Adj Close'].rolling(750).min()
        df['minall'] = df['Adj Close'] / df['Adj Close'].cummin()
        
        df['max90'] = df['Adj Close']/ df['Adj Close'].rolling(66).max()
        df['max180'] = df['Adj Close']/ df['Adj Close'].rolling(126).max()
        df['max1y'] = df['Adj Close'] / df['Adj Close'].rolling(252).max()
        df['max3y'] = df['Adj Close'] / df['Adj Close'].rolling(750).max()
        df['maxall'] = df['Adj Close'] / df['Adj Close'].cummax()
        
        df['target'] = df['Adj Close'].pct_change(periods  = -future_period_tbpredict).astype(float)         
        df['target'] =  (1/ (df['target'] + 1) -1 )
        
        # df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        # df = df.dropna(how='any',axis=0) 
        df.dropna(subset=[n for n in df if n != 'target'], inplace=True)
        df.reset_index(level=0, inplace=True)
        del df['High']
        #del df['Close']
        del df['Low']
        del df['Open']
        del df['Volume']    
        del df['EMA_5']
        del df['EMA_15']
        del df['SMA_50']
        del df['SMA_100']
        del df['SMA_200']
        df = df.set_index(['stock','Date', 'weekday','subsector','Adj Close',
                           'HLspread', 'COC_spread', 'CLC_spread', 'Gap','Volume_Rate',
                           '1HLspread','1COC_spread', '1CLC_spread', '1Gap','1Volume_Rate',
                           '2HLspread','2COC_spread', '2CLC_spread', '2Gap','2Volume_Rate',
                           '3HLspread','3COC_spread', '3CLC_spread', '3Gap','3Volume_Rate',
                           '4HLspread','4COC_spread', '4CLC_spread', '4Gap','4Volume_Rate',
                           'EMA_5/EMA_15', 'EMA_5/SMA_100', 'EMA_5/SMA_50',
                           'EMA_5/SMA_200','EMA_15/SMA_200','SMA_50/SMA_200', 'SMA_100/SMA_200', 
                           'EMA_15/SMA_100','EMA_15/SMA_50', 'RSI_5','RSI_14', 'RSI_30', 'RSI_50',
                           '10day_return','monthly_return','2monthly_return','3monthly_return',
                           '6monthly_return','12monthly_return','24monthly_return',
                           '36monthly_return','60monthly_return',
                           'monthly_volume_pct_change','3months_volume_pct_change','yearly_volume_pct_change','24month_volume_pct_change',
                           '36month_volume_pct_change','60month_volume_pct_change',
                           'std90', 'std180','std1y','std3y',
                           'min90','min180','min1y','min3y','minall',
                           'max90','max180','max1y','max3y','maxall','target'])
        df = df.reset_index()
        # df['Target']= df.iloc[:, 71:].mean(axis=1)
        # df = df[df.columns.drop(list(df.filter(regex='_Days_return')))]   
        return df
    except RemoteDataError:
        pass
    except ValueError:
        pass
    except AttributeError:
        pass
    except KeyError:
        pass
    except:
        pass
    
def findMinVariancePort(meanReturns, covMatrix):
    try:
        number_of_assets = len(meanReturns)
        args = (meanReturns, covMatrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple( (0,1) for asset in range(number_of_assets))
        optimals = sco.minimize(getPortfolioVol, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimals
    except:
        pass
    
    
def findMaxSharpeRatioPort(meanReturns, covMatrix, riskFreeRate):
    try:
        number_of_assets = len(meanReturns)
        args = (meanReturns, covMatrix, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limits = tuple( (0,1) for asset in range(number_of_assets))
        optimals = sco.minimize(negativeSharpeRatio, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=limits, constraints=constraints)
        return optimals
    except:
        pass

def create_weekly_candle_dataset(list_of_stocks, future_days,startdate,enddate):
    column_names = ['stock','Date', 'weekday','subsector','Adj Close',
                           'HLspread', 'COC_spread', 'CLC_spread', 'Gap','Volume_Rate',
                           '1HLspread','1COC_spread', '1CLC_spread', '1Gap','1Volume_Rate',
                           '2HLspread','2COC_spread', '2CLC_spread', '2Gap','2Volume_Rate',
                           '3HLspread','3COC_spread', '3CLC_spread', '3Gap','3Volume_Rate',
                           '4HLspread','4COC_spread', '4CLC_spread', '4Gap','4Volume_Rate',
                           'EMA_5/EMA_15', 'EMA_5/SMA_100', 'EMA_5/SMA_50',
                           'EMA_5/SMA_200','EMA_15/SMA_200','SMA_50/SMA_200', 'SMA_100/SMA_200', 
                           'EMA_15/SMA_100','EMA_15/SMA_50', 'RSI_5','RSI_14', 'RSI_30', 'RSI_50',
                           '10day_return','monthly_return','2monthly_return','3monthly_return',
                           '6monthly_return','12monthly_return','24monthly_return',
                           '36monthly_return','60monthly_return',
                           'monthly_volume_pct_change','3months_volume_pct_change','yearly_volume_pct_change','24month_volume_pct_change',
                           '36month_volume_pct_change','60month_volume_pct_change',
                           'std90', 'std180','std1y','std3y',
                           'min90','min180','min1y','min3y','minall',
                           'max90','max180','max1y','max3y','maxall','target']
    data = pd.DataFrame(columns = column_names)
    data = data.set_index(['stock','Date', 'weekday','subsector','Adj Close',
                           'HLspread', 'COC_spread', 'CLC_spread', 'Gap','Volume_Rate',
                           '1HLspread','1COC_spread', '1CLC_spread', '1Gap','1Volume_Rate',
                           '2HLspread','2COC_spread', '2CLC_spread', '2Gap','2Volume_Rate',
                           '3HLspread','3COC_spread', '3CLC_spread', '3Gap','3Volume_Rate',
                           '4HLspread','4COC_spread', '4CLC_spread', '4Gap','4Volume_Rate',
                           'EMA_5/EMA_15', 'EMA_5/SMA_100', 'EMA_5/SMA_50',
                           'EMA_5/SMA_200','EMA_15/SMA_200','SMA_50/SMA_200', 'SMA_100/SMA_200', 
                           'EMA_15/SMA_100','EMA_15/SMA_50', 'RSI_5','RSI_14', 'RSI_30', 'RSI_50',
                           '10day_return','monthly_return','2monthly_return','3monthly_return',
                           '6monthly_return','12monthly_return','24monthly_return',
                           '36monthly_return','60monthly_return',
                           'monthly_volume_pct_change','3months_volume_pct_change','yearly_volume_pct_change','24month_volume_pct_change',
                           '36month_volume_pct_change','60month_volume_pct_change',
                           'std90', 'std180','std1y','std3y',
                           'min90','min180','min1y','min3y','minall',
                           'max90','max180','max1y','max3y','maxall','target'])
    data = data.reset_index()
    for stock in list_of_stocks[0:]:
        try:
            data2 = candle_dataset(stock,future_days,startdate,enddate)
            data = data.append(data2)
            print (stock)
        except RemoteDataError:
            pass
        except ValueError:
            pass
        except AttributeError:
            pass
        except KeyError:
            pass
        except:
            pass
    return data

def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    try:
        p_ret, p_var = calcPortfolioPerformance(weights, meanReturns, covMatrix)
        return -(p_ret - riskFreeRate) / p_var
    except:
        pass


def prod_predict_NN(list_of_stocks, end):
    try:
        data = create_week_candle_prod(list_of_stocks)
        #most_recent_date_data = data['Date'].max()
        fed_data = get_fed_date()
        #most_recent_date = fed_data['DATE'].max()
        data = data[data['Date'] == end]
        temp = data.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )
        temp = temp.sort_values('Date', ascending = True).fillna(method = 'ffill')
        temp = temp.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        list_of_stocks = temp['stock'].tolist()
        final = temp.loc[:, temp.columns != 'stock']
        final = final.loc[:, final.columns != 'Date']
        final = final.loc[:, final.columns != 'DATE']
        X = final.loc[:, :].values
        model_classifier_monthly = load_model('model_classifier.h5')
        # summarize model.
        norm_scaler = load(open('norm_scaler.pkl', 'rb'))
        #standard_scaler = load(open('standard_scaler.pkl', 'rb'))
        #X = standard_scaler.transform(X)
        X = norm_scaler.transform(X)
        #model_classifier_monthly = lgb.Booster(model_file='project_model.txt')
        y_pred_monthly = model_classifier_monthly.predict(X)
        y_pred_monthly = y_pred_monthly.tolist()
        y_pred_monthly = [item for sublist in y_pred_monthly for item in sublist]
        df = pd.DataFrame({'stock':list_of_stocks, 'return':y_pred_monthly})
        df = df[df['return'].between(0.0, 10)]
        #df = df.sort_values(by='y_pred_monthly', ascending=False)
        df.to_csv('Archive\\' +  str(end)[0:10] + '.csv', index=False)
        return df
    except:
        pass

def portfolio_optimizer(number_of_top_stocks, end):
    try:
        start = '2000-01-01'
        list_of_stocks1 = pd.read_csv('Archive\\' +  str(end)[0:10] + '.csv', index_col=False)
        list_of_stocks1 = list_of_stocks1.sort_values(by='return', ascending=False)
        list_of_stocks = list_of_stocks1['stock'].head(number_of_top_stocks).to_list()
        data = yf.download(list_of_stocks)['Adj Close']
        data = data.loc[start:end]   
        
        #Calculate stock mean-variance
        riskFreeRate, dur = 0.04 , 22
        windowedData = data[::dur]
        rets = np.log(windowedData/windowedData.shift(1))    
        monthlyReturn = pd.read_csv('Archive\\' +  str(end)[0:10] + '.csv', index_col=False)
        monthlyReturn = monthlyReturn.sort_values(by='return', ascending=False)
        monthlyReturn = monthlyReturn.head(number_of_top_stocks)
        series1 = monthlyReturn.iloc[:,:1].values.tolist()
        series2 = np.log(monthlyReturn.iloc[:,1:].values.tolist())
        series1 = [val for sublist in series1 for val in sublist]
        series2 = [val for sublist in series2 for val in sublist]
        meanDailyReturn = pd.Series(series2,index=pd.Index(series1, name='index'))
        covariance = rets.cov()
        weights = np.random.random(len(series1))
        weights /= np.sum(weights)
        
        minVar = findMinVariancePort(meanDailyReturn, covariance)  
        minVar_portfolio =pd.DataFrame(list(zip(list_of_stocks,minVar['x'])),
                                        columns=['stocks','weights']).sort_values(by='weights', ascending=False)
        
        maxSharpe = findMaxSharpeRatioPort(meanDailyReturn, covariance, riskFreeRate)
        maxSharpe_portfolio =pd.DataFrame(list(zip(list_of_stocks,maxSharpe['x'])),
                                          columns=['stocks','weights']).sort_values(by='weights', ascending=False)
        
        minVar_portfolio.to_csv('portfolios\\'+end+'minVar_portfolio.csv', index=False)  
        maxSharpe_portfolio.to_csv('portfolios\\'+end+'maxSharpe_portfolio.csv', index=False)
        
        return minVar_portfolio, maxSharpe_portfolio
    except:
        pass

def calcPortfolioPerformance(weights, meanReturns, covarianceMatrix):
    try:
        portReturn = np.sum( meanReturns*weights )
        portStdDev = np.sqrt(np.dot(weights.T, np.dot(covarianceMatrix, weights)))
        return portReturn, portStdDev
    except:
        pass
    
    
def performance(test_portfolio, portfolio_date,initial_capital):
    try:
        portfolio_date = pd.to_datetime(portfolio_date, format='%Y-%m-%d')
        list_of_stocks = test_portfolio['stocks'].to_list()
        perf_data = pd.DataFrame(columns=['Stock','Date', 'Close'])
        for stock in list_of_stocks:
            df = yf.download(stock)
            df['Stock'] = stock
            df = df.reset_index()
            df = df[['Stock','Date','Adj Close']]
            df.rename(columns = {'Adj Close':'Close'}, inplace = True)  
            df = df.copy()
            perf_data = perf_data.append(df)
        most_recent_date = yf.download('AAPL')
        most_recent_date = most_recent_date.reset_index()
        most_recent_date = most_recent_date[(most_recent_date['Date'] > portfolio_date)]
        most_recent_date = most_recent_date['Date']
        most_recent_date = most_recent_date.sort_values(ascending=True)
        most_recent_date = most_recent_date.head(22)
        most_recent_date = most_recent_date.max()
        perf_data_current = perf_data[(perf_data['Date'] == most_recent_date)]
        perf_data_current = perf_data_current.copy()
        perf_data_past = perf_data[(perf_data['Date'] == portfolio_date)]
        temp = test_portfolio.merge(perf_data_past,how = 'inner', left_on = ['stocks'], right_on = ['Stock'] )
        temp = temp[['stocks', 'weights','Close']]
        temp['weights'] = temp['weights'] * initial_capital
        temp['# of stocks'] = (temp['weights']/ temp['Close']).astype('int64')
        temp = temp.rename(columns={'Close':'Buying price'})
        temp = temp.merge(perf_data_current,how = 'inner', left_on = ['stocks'], right_on = ['Stock'] )
        temp['current Price'] = temp['Close']
        temp['Initial_Value'] = temp['# of stocks'] * temp['Buying price']
        temp['Current_Value'] = temp['# of stocks'] * temp['current Price']*0.998
        temp['Performance'] = ((temp['Current_Value'] / temp['Initial_Value']-1)*100).round(2)
        portfolio_performance = round(((temp['Current_Value'].sum() / temp['Initial_Value'].sum())-1)*100,4)
        del temp['stocks'],temp['Close'],temp['weights'], temp['Date']
        front_data = temp.set_index(['Stock','# of stocks', 'Buying price','current Price','Initial_Value','Current_Value','Performance'])
        front_data = front_data.reset_index()
        df_performance = pd.DataFrame([['Total_performance',0,0.00,0.00,temp['Initial_Value'].sum(),temp['Current_Value'].sum(),portfolio_performance ]], columns=['Stock','# of stocks', 'Buying price','current Price','Initial_Value','Current_Value','Performance'])
        front_data = front_data.append(df_performance)
        front_data['Initial_Value'] = front_data['Initial_Value'].map("${:,.0f}".format)
        front_data['Current_Value'] = front_data['Current_Value'].map("${:,.0f}".format)
        front_data['current Price'] = front_data['current Price'].map("${:,.0f}".format)
        front_data['Buying price'] = front_data['Buying price'].map("${:,.0f}".format)
        front_data = front_data.dropna(how='any',axis=0)
        front_data['Performance'] = np.where( front_data['Performance'] < 0.000, '-%' + front_data['Performance'].astype(str).str[1:], '%' + front_data['Performance'].astype(str))
        return front_data
    except:
        pass




