import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import scipy.stats as sst
import matplotlib.pyplot as plt
import yfinance as yf

yf.pdr_override()

tickers = ['AAPL', 'MSFT', 'NVDA', 'BABA', '^TNX', 'GOLD', 'DAL', 'PFE', 'BAC', 'DIS']
stocklist = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'BABA': 'Alibaba', '^TNX': 'Treas Yld Index',
             'GOLD': 'Barrick', 'DAL': 'Delta', 'PFE': 'Pfizer', 'BAC': 'Bank of America', 'DIS': 'Walt Disney'}
weight = [0.17070402, 0.26714263, 0.04696567, 0.00771267, 0.00833391, 0.03757462,
          0.03810221, 0.13953412, 0.21689562, 0.06703453]

# read data we got previously and change the scale
price = pd.DataFrame()
for i in tickers:
    price[i] = pdr.get_data_yahoo(i, start='2017-3-11', end='2020-3-11')['Adj Close']
sec_returns1 = ((price / price.shift(1)) - 1)

price = pd.DataFrame()
for i in tickers:
    price[i] = pdr.get_data_yahoo(i, start='2020-3-11', end='2023-3-11')['Adj Close']
sec_returns2 = ((price / price.shift(1)) - 1)

# compute the portfolio's return
portfolio = pd.DataFrame()
portfolio['before'] = np.dot(sec_returns1, weight)
portfolio['after'] = pd.Series(np.dot(sec_returns2, weight))
portfolio = portfolio.dropna()

# Descriptive Statistical Analysis
portfolio.describe()
# plot the portfolio's price line plot
portfolio['before'].plot()
portfolio['after'].plot()
plt.legend()
plt.savefig('portfolio_return comparison.png')
plt.show()

# h-test, we want to see whether our portfolio's mean is:
# equal before and after 2020-1-1  --(null hypothesis)
# or not -- (Alternative hypothesis)
p_value = sst.ttest_1samp(portfolio['before'], popmean=0.000, nan_policy='omit')
print(f'mean before mean = 0 ttest result: {p_value}')

p_value = sst.ttest_1samp(portfolio['after'], popmean=0.000, nan_policy='omit')
print(f'mean after mean = 0 result: {p_value}')

p_value = sst.ttest_rel(a=portfolio['before'], b=portfolio['after'], nan_policy='omit')
print(f'mean before = after result: {p_value}')

# h-test, we want to see whether our portfolio's var is:
# equal before and after 2020-1-1  --(null hypothesis)
# or not -- (Alternative hypothesis)
F_stat = max(portfolio['before'].var() / portfolio['after'].var(), portfolio['after'].var() / portfolio['before'].var())
p_value = 1 - sst.f.cdf(F_stat, dfn=portfolio['after'].count() - 1, dfd=portfolio['before'].count() - 1)
print(f'var t-test result: {p_value}')
