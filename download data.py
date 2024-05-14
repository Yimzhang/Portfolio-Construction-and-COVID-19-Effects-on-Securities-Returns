import pandas as pd
import yfinance
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from pandas_datareader.famafrench import get_available_datasets

# download rf and rm
get_available_datasets()
ff = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1991-1-1')[0]
mm = pdr.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start='1991-1-1')[0]
factors = pd.merge(ff, mm, left_index=True, right_index=True)
factors.head()

yfinance.pdr_override()

price = pd.DataFrame()
volume = pd.DataFrame()
tickers = ['AAPL', 'MSFT', 'NVDA', 'BABA', '^TNX', 'GOLD', 'DAL', 'PFE', 'BAC', 'DIS']

for i in tickers:
    price[i] = pdr.get_data_yahoo(i, start='1991-1-1')['Adj Close']
    volume[i] = pdr.get_data_yahoo(i, start='1991-1-1')['Volume']
price.head()

sec_returns = ((price / price.shift(1)) - 1) * 100
sec_returns = sec_returns.dropna()
sec_returns.head()

total = pd.merge(factors, sec_returns, left_index=True, right_index=True)
total = total.dropna()
total.head()

writer1 = pd.ExcelWriter('merge_data.xlsx')
writer2 = pd.ExcelWriter('data.xlsx')
total.to_excel(writer1, index=False)
sec_returns.to_excel(writer2, index=False)
writer1.save()
writer2.save()

# data discribe
total.describe()

# visualize
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

for i in price:  # price line chart
    price[i].plot()
    plt.ylabel(f'{i} Price - $')
    plt.savefig(f'{i}.png')
    plt.show()
    plt.close('all')

for i in volume:  # volume line chart
    volume[i].plot()

plt.savefig('volume_line.png')
plt.show()

# volume bar chart
volume = volume.drop('^TNX', axis=1)
for i in volume:
    plt.bar(i, volume[i].mean())

plt.savefig('volume_bar.png')
plt.show()
