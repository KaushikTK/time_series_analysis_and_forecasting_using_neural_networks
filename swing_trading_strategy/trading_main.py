import os
import yfinance as yf
import numpy as np
import pandas_ta as ta
import numpy as np
import pandas as pd

def trade_donchian(row):
    global trades, trade_open
    row = row.to_dict()

    if((trade_open==True) and (row['long'] == 1)):
        _trade = trades[-1]
        if(float(_trade['swing_low'])>float(row['Close'])): _trade['swing_low'] = row['Close']
        if(float(_trade['swing_low'])>float(row['Adj Close'])): _trade['swing_low'] = row['Adj Close']
        if(float(_trade['swing_high'])<float(row['Close'])): _trade['swing_high'] = row['Close']
        if(float(_trade['swing_high'])<float(row['Adj Close'])): _trade['swing_high'] = row['Adj Close']
        trades[-1] = _trade
        del _trade
        
    elif((trade_open==False) and (row['long'] == 1)):
        # open trade
        trade_open = True
        _trade = {
            'buy_date': row['next_date'],
            'swing_high': round(row['next_day_open_price']*1.005,2),
            'swing_low': round(row['next_day_open_price']*1.005,2),
            'buy_price': round(row['next_day_open_price']*1.005,2),
            'sell_price': None,
            'sell_date': None,
        }
        trades.append(_trade)
        del _trade
        
    elif((trade_open==False) and (row['short'] == 1)): pass
        
    elif((trade_open==True) and (row['short'] == 1)):
        # close trade
        trade_open = False
        _trade = trades[-1]
        _trade['sell_date'] = row['next_date']
        _trade['sell_price'] = round(row['next_day_open_price']*0.995,2)

        if(float(_trade['swing_low'])>float(row['Close'])): _trade['swing_low'] = row['Close']
        if(float(_trade['swing_low'])>float(row['Adj Close'])): _trade['swing_low'] = row['Adj Close']
        if(float(_trade['swing_high'])<float(row['Close'])): _trade['swing_high'] = row['Close']
        if(float(_trade['swing_high'])<float(row['Adj Close'])): _trade['swing_high'] = row['Adj Close']
            
        trades[-1] = _trade
        del _trade


def backtest(stock, period, low=20, high=20):
    global trades, trade_open
    
    data = yf.download(f'{stock}.ns', period=period, progress=False).reset_index()
    
    data['week_day'] = pd.to_datetime(data['Date']).dt.day_name()
    
    data = data[~data['week_day'].isin(['Saturday', 'Sunday'])]
    
    data[['low', 'mid', 'high']] = data.ta.donchian(lower_length=low, upper_length=high)
    
    data['long'] = ((data['Close']==data['low'])|(data['Low']==data['low'])).astype('int')
    data['short'] = ((data['Close']==data['high'])|(data['High']==data['high'])).astype('int')
    
    data['next_day_open_price'] = data['Open'].shift(-1)
    data['next_date'] = data['Date'].shift(-1).astype('string')
    
    trade_open = False
    
    trades = []
    
    data.dropna(inplace=True)
    
    cols = ['Date', 'Open', 'Close', 'Adj Close', 'Low', 'High', 'low', 'mid', 'high', 'long', 'short', 'next_day_open_price', 'next_date']
    data = data[cols]
    data.apply(trade_donchian, axis=1)

    if(len(trades)==0): return None
    x = pd.DataFrame(trades)
    
    x['buy_date'] = pd.to_datetime(x['buy_date'], format="%Y-%m-%d", dayfirst=True)
    x['sell_date'] = pd.to_datetime(x['sell_date'], format="%Y-%m-%d", dayfirst=True)
    x['returns'] = round(100*(x['sell_price']-x['buy_price'])/x['buy_price'],2)
    x['holding_period'] = (x['sell_date'] - x['buy_date']).dt.days
    x['stock'] = stock
    return x


file_name = 'n200'
TRADES = pd.DataFrame()
trades = []
trade_open = False

stocks_list = [i.upper() for i in pd.read_csv(f'./index_stocks/{file_name}.csv')['Symbol'].values if 'adani' not in i.lower()]

for stock in stocks_list:
    _tr = backtest(stock, '10y', 20, 20)
    if(len(TRADES)==0): TRADES = _tr
    else: TRADES = pd.concat([TRADES, _tr], ignore_index=True)

TRADES.to_csv('trades.csv', index=False)

TRADES = TRADES.dropna(how='any')

TRADES['positive'] = 0
TRADES.loc[TRADES['returns']>0, 'positive'] = 1

TRADES = TRADES.groupby('stock').agg({'returns': ['sum', 'mean'], 'positive':['sum','count']}).reset_index()
TRADES['proba'] = TRADES['positive']['sum']/TRADES['positive']['count']

# condition 1 => proba >= 0.65
TRADES = TRADES[TRADES['proba']>=0.65]

#condition 2 => returns sum > 0 & mean > 0
TRADES = TRADES[TRADES['returns']['sum']>0]
TRADES = TRADES[TRADES['returns']['mean']>0]

TRADES['Symbol'] = TRADES['stock']

TRADES['Symbol'].to_csv('./index_stocks/stock_universe.csv', index=False)

os.system('filter_stocks.bat')