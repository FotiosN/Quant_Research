# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:06:06 2021

@author: fotis
"""

#import Heuristics as y
import urllib.request
import data_engine as daf
import pandas as pd
import os
import re
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import concurrent.futures
from datetime import datetime, timedelta, date
import requests, zipfile, io
from pandas_datareader._utils import RemoteDataError
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")

def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False
    
def download(new_list):
    for i in new_list:
        x = os.path.exists('.\\EDGAR_archive\\'+ i)
        if x == False:  
            dest_folder = '.\\EDGAR_archive\\'
            url = 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/' + (i) 
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)  # create folder if it does not exist
        
            filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
            file_path = os.path.join(dest_folder, filename)
        
            r = requests.get(url, stream=True)
            if r.ok:
                print("saving to", os.path.abspath(file_path))
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
            else:  # HTTP status code 4XX/5XX
                print("Download failed: status code {}\n{}".format(r.status_code, r.text))
        else:
            continue



def add_new_data(new_list): 
    new_dataset = pd.read_csv('pricing_data\\book_data.csv')
    for i in new_list:
        x = os.path.exists('.\\EDGAR_archive\\'+ i + '.csv')
        if x == False:       
            url = 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/' + (i) 
            if url_is_alive(url)==True:
                cik = pd.read_csv('sec_files\\cik.csv')
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                sub = pd.read_table(z.open('sub.tsv'))
                num = pd.read_table(z.open('num.tsv'))
                num = num.loc[num['dimh'].isin(['0x00000000'])]
                num = num.loc[num['tag'].isin(['StockholdersEquity','Assets','CashAndCashEquivalentsAtCarryingValue',
                'LiabilitiesAndStockholdersEquity','CommonStockSharesOutstanding',
                'Liabilities','AssetsCurrent','LiabilitiesCurrent'
                'IntangibleAssetsNetExcludingGoodwill','ResearchAndDevelopmentExpense',
                'Goodwill','ProfitLoss', 'NetIncomeLoss','GrossProfit','DebtInstrumentInterestRateStatedPercentage',
                'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                'AccountsPayableCurrent','InventoryNet','RealEstateGrossAtCarryingValue',
                'AccountsReceivableNetCurrent','OperatingIncomeLoss','Revenues',
                'SalesRevenueNet','PropertyPlantAndEquipmentGross','OperatingExpenses','EarningsPerShareBasic'
                ])]            
                result = pd.merge(num, sub, how="inner", on=["adsh"])
                del sub, num, r, z
                final = pd.merge(result, cik, how="inner", on=["cik"])
                del result,cik, url
            try:
                final['ddate'] = pd.to_datetime(final['ddate'], format='%Y%m%d', errors='coerce')
                final = final.dropna(subset = ['ddate'])
                final['period'] = pd.to_datetime(final['period'], errors='coerce')
                final = final.dropna(subset = ['period'])
                final['accepted'] = pd.to_datetime(final['accepted'], errors='coerce')
                final = final.dropna(subset = ['accepted'])
                #final = final.dropna()
            except RemoteDataError as r:
                pass
                print(r)
            except ValueError as v:
                print(v)
                pass
            except TypeError as t:
                print (t)
                pass
            final = final[['Ticker','period', 'ddate','accepted','form','tag','qtrs','value']]
            final = final.loc[final['form'].isin(['10-K','10-Q'])]
            new_dataset = new_dataset.append(final, sort=False)
            del final
            new_dataset['ddate'] = new_dataset["ddate"].values.astype('datetime64[D]')
            new_dataset['period'] = new_dataset["period"].values.astype('datetime64[D]')
            new_dataset['accepted'] = new_dataset["accepted"].values.astype('datetime64[D]')
            new_dataset = new_dataset.sort_values('accepted', ascending = True).drop_duplicates(['Ticker','tag','form','ddate','qtrs','period'],keep = 'last')
            print(i)
        else:
            continue
    new_dataset.to_csv('pricing_data\\book_data.csv', index=False)
    return new_dataset


# form = 10-K, 10-Q
#new_list = list with the EDGAR file names eg ['2011q3_notes.zip','2011q4_notes.zip']
def get_local_data(new_list): 
    new_dataset = pd.DataFrame(columns=['Ticker','period', 'ddate','accepted','form','tag','qtrs','value'])  
    for i in new_list:
        cik = pd.read_csv('sec_files\\cik.csv')
        z = zipfile.ZipFile('.\\EDGAR_archive\\'+ i, 'r')
        sub = pd.read_table(z.open('sub.tsv'))
        num = pd.read_table(z.open('num.tsv'))
        num = num.loc[num['dimh'].isin(['0x00000000'])]
        num = num.loc[num['tag'].isin(['StockholdersEquity','Assets','CashAndCashEquivalentsAtCarryingValue',
        'LiabilitiesAndStockholdersEquity','CommonStockSharesOutstanding',
        'Liabilities','AssetsCurrent','LiabilitiesCurrent'
        'IntangibleAssetsNetExcludingGoodwill','ResearchAndDevelopmentExpense',
        'Goodwill','ProfitLoss', 'NetIncomeLoss','GrossProfit','DebtInstrumentInterestRateStatedPercentage',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'AccountsPayableCurrent','InventoryNet','RealEstateGrossAtCarryingValue',
        'AccountsReceivableNetCurrent','OperatingIncomeLoss','Revenues',
        'SalesRevenueNet','PropertyPlantAndEquipmentGross','OperatingExpenses','EarningsPerShareBasic'
        ])]
        result = pd.merge(num, sub, how="inner", on=["adsh"])
        del sub, num, z
        final = pd.merge(result, cik, how="inner", on=["cik"])
        del result,cik
        try:
            final['ddate'] = pd.to_datetime(final['ddate'], format='%Y%m%d', errors='coerce')
            final = final.dropna(subset = ['ddate'])
            final['accepted'] = pd.to_datetime(final['accepted'], errors='coerce')
            final = final.dropna(subset = ['accepted'])
            # final = final.dropna()
            final['period'] = pd.to_datetime(final['period'], format='%Y%m%d', errors='coerce')
            final = final.dropna(subset = ['period'])
        except RemoteDataError as r:
            pass
            print(r)
        except ValueError as v:
            print(v)
            pass
        except TypeError as t:
            print (t)
            pass
        final = final[['Ticker','period', 'ddate','accepted','form','tag','qtrs','value']]
        new_dataset = new_dataset.append(final, sort=False)
        del final
        new_dataset['ddate'] = new_dataset["ddate"].values.astype('datetime64[D]')
        new_dataset['period'] = new_dataset["period"].values.astype('datetime64[D]')
        new_dataset['accepted'] = new_dataset["accepted"].values.astype('datetime64[D]')
        new_dataset = new_dataset.sort_values('accepted', ascending = True).drop_duplicates(['Ticker','tag','form','ddate','qtrs','period'],keep = 'last')
        print(i)
    new_dataset.to_csv('pricing_data\\book_data.csv', index=False)
    cleaning_dataset()

def create_date_data(startdate,enddate):
    datee = pd.date_range(start=startdate,end=enddate)
    datee = datee.to_frame(index=False)
    datee.rename(columns = {0:'Date'}, inplace = True)
    return datee

def process_per(accounting_data, date_data, stock,tags):
    data = date_data
    for acc_tag in tags:
        xi = accounting_data.loc[(accounting_data['tag'] == acc_tag) & (accounting_data['Ticker'] == stock)]
        final = date_data.merge(xi,how = 'outer', left_on = ['Date'], right_on = ['accepted'] )
        final = final.sort_values('Date', ascending = True).fillna(method = 'ffill')
        final.rename(columns = {'value':acc_tag}, inplace = True)
        final = final.drop(columns = ['Ticker', 'tag', 'form','qtrs', 'accepted'])
        data = data.merge(final,how = 'inner', left_on = ['Date'], right_on = ['Date'] )
        data = data.sort_values('Date', ascending = True).drop_duplicates(['Date'],keep = 'first')
        del final,xi
    return data

def Balance_sheet(stock,startdate, enddate):
    date_data = create_date_data(startdate,str(date.today()))
    accounting_data = pd.read_csv('pricing_data\\book_data.csv')
    accounting_data['accepted'] = pd.to_datetime(accounting_data['accepted']) 
    tags = ['StockholdersEquity','Assets','CashAndCashEquivalentsAtCarryingValue',
            'LiabilitiesAndStockholdersEquity','CommonStockSharesOutstanding',
            'Liabilities','AssetsCurrent','LiabilitiesCurrent',
            'IntangibleAssetsNetExcludingGoodwill','ResearchAndDevelopmentExpense',
            'Goodwill','ProfitLoss', 'NetIncomeLoss','GrossProfit','DebtInstrumentInterestRateStatedPercentage',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'AccountsPayableCurrent','InventoryNet','RealEstateGrossAtCarryingValue',
            'AccountsReceivableNetCurrent','OperatingIncomeLoss','Revenues',
            'SalesRevenueNet','PropertyPlantAndEquipmentGross','OperatingExpenses','EarningsPerShareBasic']
    yolo = process_per(accounting_data, date_data, stock,tags)
    yolo = yolo[-yolo["Date"].isin([enddate.strftime("%Y-%m-%d")])]
    scalar = int(yolo['CommonStockSharesOutstanding'].tail(1))
    for lolo in tags:
        yolo['check'] = 1/ (1 - abs(yolo[lolo].pct_change(periods  = 1).astype(float)))
        pipi = yolo[yolo['check'] > 1000][lolo].tolist()
        for i in pipi:
            i = int(i)
            yolo[lolo] = yolo[lolo].replace(i, np.nan)
        del yolo['check']
        yolo[lolo]=yolo[lolo]/scalar
    yolo = yolo.fillna(method = 'ffill')
    del yolo['CommonStockSharesOutstanding']
    return yolo

def cleaning_dataset():
    new_dataset = pd.read_csv('pricing_data\\book_data.csv')
    new_dataset['accepted'] = pd.to_datetime(new_dataset['accepted'])
    new_dataset['period'] = pd.to_datetime(new_dataset['period'])
    new_dataset['ddate'] = pd.to_datetime(new_dataset['ddate'])
    new_dataset['check1'] = new_dataset['accepted'] - new_dataset['period']
    new_dataset['check2'] = new_dataset['accepted'] - new_dataset['ddate']
    new_dataset['check1'] = new_dataset['check1'].dt.days
    new_dataset['check2'] = new_dataset['check2'].dt.days
    new_dataset = new_dataset[(new_dataset['check1'] <= 60) & (new_dataset['check1'] >= 0) & (new_dataset['check2'] <= 60) & (new_dataset['check2'] >= 0)]
    del new_dataset['check1'],new_dataset['check2'],new_dataset['period'],new_dataset['ddate']
    new_dataset['ddate'] = new_dataset["ddate"].values.astype('datetime64[D]')
    new_dataset['period'] = new_dataset["period"].values.astype('datetime64[D]')
    new_dataset['accepted'] = new_dataset["accepted"].values.astype('datetime64[D]')
    new_dataset = new_dataset.sort_values('value', ascending = True).drop_duplicates(['Ticker','tag','form','ddate','qtrs','period'],keep = 'last')
    del new_dataset['AccountsPayableCurrent'], new_dataset['InventoryNet'], new_dataset['OperatingIncomeLoss'], new_dataset['AccountsReceivableNetCurrent']
    new_dataset.to_csv('pricing_data\\book_data.csv', index=False)
    return new_dataset

##################################################################################################################################
new_list = [
    # '2021_03_notes.zip', '2021_02_notes.zip', '2021_01_notes.zip',
    #         '2021_04_notes.zip', '2021_05_notes.zip', '2021_06_notes.zip',
    #         '2021_07_notes.zip', '2021_08_notes.zip',
    #         '2020_12_notes.zip', '2020_11_notes.zip', '2020_10_notes.zip',
    #         '2020q3_notes.zip', '2020q2_notes.zip', '2020q1_notes.zip',
    #         '2019q4_notes.zip', '2019q3_notes.zip', '2019q2_notes.zip', '2019q1_notes.zip',
    #         '2018q4_notes.zip', '2018q3_notes.zip', '2018q2_notes.zip', '2018q1_notes.zip',
    #         '2017q4_notes.zip', '2017q3_notes.zip', '2017q2_notes.zip', '2017q1_notes.zip',
    #         '2016q4_notes.zip', '2016q3_notes.zip', '2016q2_notes.zip', '2016q1_notes.zip',
    #         '2015q4_notes.zip', '2015q3_notes.zip', '2015q2_notes.zip', '2015q1_notes.zip',
            '2014q4_notes.zip', '2014q3_notes.zip', '2014q2_notes.zip', '2014q1_notes.zip',
            '2013q4_notes.zip', '2013q3_notes.zip', '2013q2_notes.zip', '2013q1_notes.zip',
            '2012q4_notes.zip', '2012q3_notes.zip', '2012q2_notes.zip', '2012q1_notes.zip',
            '2011q4_notes.zip', '2011q3_notes.zip', '2011q2_notes.zip', '2011q1_notes.zip',
            '2010q4_notes.zip', '2010q3_notes.zip'
            ]

#data = get_local_data(new_list)

# new_list = ['2021_04_notes.zip']
# add_new_data(new_list)
# download(new_list)


days_in_past = 6000
startdate = datetime.today() - timedelta(days=days_in_past)
startdate = startdate.strftime("%Y-%m-%d")
enddate = date.today()
stock = 'AMD'

Balance_sheet1 = Balance_sheet(stock,startdate, enddate)

fed_data = daf.get_fed_date()
pricing_data = daf.create_weekly_candle_dataset([stock],30,startdate,enddate)
pricing_data = pricing_data.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )

#Balance_sheet = Balance_sheet.merge(fed_data,how = 'inner', left_on = ['Date'], right_on = ['DATE'] )
accounting_data = Balance_sheet1.merge(pricing_data,how = 'inner', left_on = ['Date'], right_on = ['Date'] )
pricing_data = accounting_data.sort_values('Date', ascending = True).fillna(method = 'ffill')
conditions = pricing_data.StockholdersEquity.isna()
pricing_data = pricing_data[~conditions]


del pricing_data['DATE']

