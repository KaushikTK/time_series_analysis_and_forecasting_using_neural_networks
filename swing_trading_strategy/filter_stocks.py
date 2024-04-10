import yfinance as yf
import numpy as np
import pandas_ta as ta
import pandas as pd
from datetime import date
from pytz import timezone 
from datetime import datetime


file_name = 'stock_universe'

stocks_list = [i.upper() for i in pd.read_csv(f'./index_stocks/stock_universe.csv')['Symbol'].values if 'adani' not in i.lower()]

LONG = []
SHORT = []

for stock in stocks_list:
    data = yf.download(f'{stock}.ns', period='3mo', progress=False).reset_index()
    
    data['week_day'] = pd.to_datetime(data['Date']).dt.day_name()
    
    # data = data[~data['week_day'].isin(['Saturday', 'Sunday'])]
    
    data[['low', 'mid', 'high']] = data.ta.donchian(lower_length=20, upper_length=20)
    
    data['long'] = ((data['Close']==data['low'])|(data['Low']==data['low'])).astype('int')
    data['short'] = ((data['Close']==data['high'])|(data['High']==data['high'])).astype('int')

    short = data[(data['short']>0)].sort_values(by=['Date'])
    long = data[data['long']>0].sort_values(by=['Date'])

    short['curr_date'] = pd.to_datetime((date.today()))
    long['curr_date'] = pd.to_datetime(str(date.today()))

    short_flag = short[short['Date']==short['curr_date']].shape[0]
    long_flag = long[long['Date']==long['curr_date']].shape[0]

    if(short_flag>0): SHORT.append(stock)
    if(long_flag>0): LONG.append(stock)

print('LONG')
print(LONG)

print()

print('SHORT')
print(SHORT)