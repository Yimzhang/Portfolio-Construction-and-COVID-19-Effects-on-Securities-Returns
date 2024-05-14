import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.optimize as sco
import statsmodels.api as sm
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# read our portfolio's data
io = r'/Users/yimingzhang/PycharmProjects/fin240/group_project/New GW/merge_data.xlsx'
total = pd.read_excel(io)

io = r'/Users/yimingzhang/PycharmProjects/fin240/group_project/New GW/data.xlsx'
sec_returns = pd.read_excel(io)

# we use 4-factors module to compute the beta of our portfolio
tickers = ['AAPL', 'MSFT', 'NVDA', 'BABA', '^TNX', 'GOLD', 'DAL', 'PFE', 'BAC', 'DIS']

sec_beta = pd.DataFrame(np.nan, index=tickers, columns=['const', 'Mkt-RF', 'SMB', 'HML'])
for t in tickers:
    X = total[['Mkt-RF', 'SMB', 'HML', 'Mom   ']]
    X1 = sm.add_constant(X)
    Y = total[t] - total['RF']
    reg = sm.OLS(Y, X1).fit()
    sec_beta.loc[t, :] = reg.params
sec_beta

# Compute annually market return, risk-free rate
rm_minus_rf = (np.exp(np.mean(np.log(total['Mkt-RF'] / 100 + 1))) ** 252) - 1

rf = (np.exp(np.mean(np.log(total['RF'] / 100 + 1))) ** 252) - 1

# Using CAPM module to compute the theoretical annual return
expected_returns = []
for i in sec_returns:
    r_annually = (sec_beta['Mkt-RF'][i] * rm_minus_rf) + rf
    expected_returns = np.append(expected_returns, r_annually)
print(expected_returns)

# calculate the cov matrix：
cov_matrix = sec_returns.cov() * 252 / 100

# use monte carlos method to construct 20,000 portfolio, compute every portfolio's return and variation.
# plot a return-sigma scatter plot.
# To make it easier, we assume there's no short sell. Therefore, each weight vector is between 0-1
# plot a return-sigma scatter plot. Efficient frontier
# 利用蒙特卡洛模拟随机生成两万个投资组合，计算出每个组合的收益率与波动性，并在均值-标准差平面画图，
# 为使得问题简单化，我们假设市场不允许卖空，因此权重向量中每一项都位于0-1之间。
number_assets = 10
portfolio_returns = []
portfolio_sigma = []
sharpe_ratio = []
for single_portfolio in range(200000):
    weights = np.random.random(number_assets)
    weights = weights / (np.sum(weights))
    returns = np.dot(weights, expected_returns)
    sigma = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_returns.append(returns)
    portfolio_sigma.append(sigma)
    sharpe = (returns - rf) / sigma
    sharpe_ratio.append(sharpe)
portfolio_returns = np.array(portfolio_returns)
portfolio_sigma = np.array(portfolio_sigma)

plt.style.use('seaborn-dark')
plt.figure(figsize=(9, 5))
plt.scatter(portfolio_sigma, portfolio_returns)
plt.grid(True)
plt.xlabel('expected sigma')
plt.ylabel('expected return')
plt.savefig('Efficient frontier.png')
plt.show()


# we choose the portfolio with highest Sharpe-ratio, which equivalent to minimum -(sharpe_ratio).
# In that case we can use the minimization optimization algorithm sco.minimize to find the optimal portfolio.
# The boundary condition is each item of the weight needs to be between 0 and 1, and the constraint condition is that the sum of the weights is 1.
# 采用夏普率最大的投资组合作为最优投资组合，夏普率最大等价于负夏普率最小，故可以利用最小化优化算法sco.minimize找出最优投资组合。边界条件为权重向量的每一项都需要在0-1之间，约束条件为权重之和为1。
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.dot(expected_returns, weights)
    port_sigma = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (port_returns - rf) / port_sigma


def min_func_sharpe(weights):
    return -statistics(weights)

# minimization optimization algorithm sco.minimize to find the optimal portfolio
bnds = tuple((0, 1) for x in range(number_assets))  # limit weight in [0,1)
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, number_assets * [1. / number_assets, ], method='SLSQP', bounds=bnds,
                    constraints=cons)

# we print out the weight and the sharpe-ratio of the 10 assets.
weights = opts['x'].round(10)
sharpe_ratio = np.dot(opts['x'], sec_beta['Mkt-RF']).round(10)
print(f'the weights are:{weights}')
print(f'the sharpe-ratio of our portfolio is {sharpe_ratio}')
