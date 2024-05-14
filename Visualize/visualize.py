import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# visualize portfolio's price and trading volume

weight = [0.17070402, 0.26714263, 0.04696567, 0.00771267, 0.00833391, 0.03757462,
          0.03810221, 0.13953412, 0.21689562, 0.06703453]

price = pd.DataFrame()
volume = pd.DataFrame()
tickers = ['AAPL', 'MSFT', 'NVDA', 'BABA', '^TNX', 'GOLD', 'DAL', 'PFE', 'BAC', 'DIS']
for i in tickers:
    price[i] = pdr.get_data_yahoo(i, start='2017-3-11')['Adj Close']

price['portfolio'] = np.dot(price, weight)
price['portfolio'].plot()
plt.ylabel('Portfolio Price - $')
plt.savefig('portfolio_price.png')
plt.show()
plt.close('all')
