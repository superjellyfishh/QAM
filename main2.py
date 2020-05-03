import pandas as pd
from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
import numpy as np
from numpy.linalg import inv, pinv
import scipy.optimize

# import matplotlib.pyplot as plt


# %% Importing the market data
market = pd.read_excel(r'MarketData.xlsx')

# Converting the dates to a readable format
market['Dates'] = pd.to_datetime(market.Dates)

# %% Getting the asset names etc...
assetnames = market.columns[3:len(market.columns):2]
benchname = market.columns[1]
mktcapsnames = market.columns[2:len(market.columns):2]
mktcapbench = market[mktcapsnames[0]]

# %% Computing the returns...
bench_returns = market[benchname].pct_change()
# Not selecting the benchmark for now
returns = market[assetnames].pct_change()
returns = returns.drop(0)
npreturns = returns.to_numpy()

# %% Storing the market caps & computing weights...
mktcaps = market[mktcapsnames[1:]]
mktcaps["SUM"] = mktcaps.sum(axis=1)
weights = mktcaps.loc[:, mktcapsnames[1]
    :mktcapsnames[-1]].div(mktcaps["SUM"], axis=0)
weights = weights.drop(0)
npweight = weights.to_numpy()

# %% Computing the prior


def assets_meanvar(returns):
    # Calculate expected returns
    expreturns = array([])
    (rows, cols) = returns.shape
    for r in range(rows):
        expreturns = append(expreturns, mean(returns[r]))

    # Compute covariance matrix
    covars = cov(returns)
    # Annualize expected returns and covariances
    # Assumes 255 trading days per year
    expreturns = (1 + expreturns) ** 12 - 1
    covars = covars * 12

    return expreturns, covars


# Compute the expected return of the portfolio.
def compute_mean(W, R):
    return sum(R * W)


# Compute the variance of the portfolio.
def compute_var(W, C):
    return dot(dot(W, C), W)


# Combination of the two functions above - mean and variance of returns calculation.
def compute_mean_var(W, R, C):
    return compute_mean(W, R), compute_var(W, C)


def weight(data=None, P=np.array([[1, 0, 0, 0, 0, -1], [1, 0, 0, -1, 0, 0]]), Q=np.array([0.05, 0.05]), tau=0.1, omega=np.array([[0.5, 0], [0, 0.3]]), lmbda=2):

    expreturns, covars = assets_meanvar(np.transpose(npreturns))
    R = expreturns  # R is the vector of expected returns
    C = covars  # C is the covariance matrix
    rf = 0.001  # rf is the risk-free rate
    W = npweight[-1]

    new_mean = compute_mean(W, R)
    new_var = compute_var(W, C)

    gamma = 3
    Pi = dot(dot(gamma, C), W)
    sub_a = inv(dot(tau, C))
    sub_b = dot(dot(transpose(P), inv(omega)), P)
    sub_c = dot(inv(dot(tau, C)), Pi)
    sub_d = dot(dot(transpose(P), inv(omega)), Q)
    Pi_new = dot(inv(sub_a + sub_b), (sub_c + sub_d))

    WeightBL = 1 / lmbda * np.matmul(inv(C), Pi_new)
    return WeightBL


w = weight()

print(w)
# Solve for optimal portfolio weights [constrained version]

# def fitness(W, R, C, r):
#     # For given level of return r, find weights which minimizes portfolio variance.
#     mean_1, var = compute_mean_var(W, R, C)
#     # Penalty for not meeting stated portfolio return effectively serves as optimization constraint
#     # Here, r is the 'target' return
#     penalty = 0.1 * abs(mean_1 - r)
#     return var + penalty
#
#
# def solve_weights(R, C, rf):
#     n = len(R)
#     W = ones([n]) / n  # Start optimization with equal weights
#     b_ = [(0.1, 1) for i in range(n)]  # Bounds for decision variables
#     c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Constraints - weights must sum to 1
#     # 'target' return is the expected return on the market portfolio
#     optimized = scipy.optimize.minimize(fitness, W, (R, C, sum(R * W)), method='SLSQP', constraints=c_, bounds=b_)
#     if not optimized.success:
#         raise BaseException(optimized.message)
#     return optimized.x


# WeightBL = solve_weights(Pi_new + rf, C, rf)
