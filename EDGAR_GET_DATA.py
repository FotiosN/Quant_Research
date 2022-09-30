# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:28:07 2020

@author: fotis
"""

import pyodbc
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import zipfile


x = ['2011q1_notes.zip',
'2012q1_notes.zip',
'2013q1_notes.zip',
'2014q1_notes.zip',
'2015q1_notes.zip',
'2016q1_notes.zip',
'2017q1_notes.zip',
'2018q1_notes.zip',
'2019q1_notes.zip',
'2020q1_notes.zip',
'2009q2_notes.zip',
'2010q2_notes.zip',
'2011q2_notes.zip',
'2012q2_notes.zip',
'2013q2_notes.zip',
'2014q2_notes.zip',
'2015q2_notes.zip',
'2016q2_notes.zip',
'2017q2_notes.zip',
'2018q2_notes.zip',
'2019q2_notes.zip',
'2020q2_notes.zip',
'2009q3_notes.zip',
'2010q3_notes.zip',
'2011q3_notes.zip',
'2012q3_notes.zip',
'2013q3_notes.zip',
'2014q3_notes.zip',
'2015q3_notes.zip',
'2016q3_notes.zip',
'2017q3_notes.zip',
'2018q3_notes.zip',
'2019q3_notes.zip',
'2020q3_notes.zip',
'2009q4_notes.zip',
'2010q4_notes.zip',
'2011q4_notes.zip',
'2012q4_notes.zip',
'2013q4_notes.zip',
'2014q4_notes.zip',
'2015q4_notes.zip',
'2016q4_notes.zip',
'2017q4_notes.zip',
'2018q4_notes.zip',
'2019q4_notes.zip']

zf = zipfile.ZipFile('2010q1_notes.zip') 
df = pd.read_csv(zf.open('num.tsv'),delimiter ='\t', encoding = "ISO-8859-1")
df.to_csv('2010q1_notes.zip' + 'num.csv')
df3 = pd.read_csv(zf.open('sub.tsv'),delimiter ='\t', encoding = "ISO-8859-1")
df3.to_csv('2010q1_notes.zip' +'sub.csv')

for i in x:
    zf = zipfile.ZipFile(i) 
    df = pd.read_csv(zf.open('num.tsv'),delimiter ='\t', encoding = "ISO-8859-1")
    df.to_csv(i + 'num.csv')
    #df2.append(df, ignore_index=True, sort=False)
    df1 = pd.read_csv(zf.open('sub.tsv'),delimiter ='\t', encoding = "ISO-8859-1")
    df1.to_csv(i + 'sub.csv')
    #df3.append(df1, ignore_index=True, sort=False)

#df2.to_csv('num.csv')
#df3.to_csv('sub.csv')
    