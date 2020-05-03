#%% try 1
import pandas as pd
from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
import numpy as np
from numpy.linalg import inv, pinv
import scipy.optimize


# Defining some functions..
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


def weight(P, Q, tau, omega, lmbda, rf):
    market = pd.read_excel(r'MarketData.xlsx')

    # Converting the dates to a readable format
    market['Dates'] = pd.to_datetime(market.Dates)
    #  Getting the asset names etc...
    assetnames = market.columns[3:len(market.columns):2]
    benchname = market.columns[1]
    mktcapsnames = market.columns[2:len(market.columns):2]
    mktcapbench = market[mktcapsnames[0]]

    #Computing the returns...
    bench_returns = market[benchname].pct_change()
    returns = market[assetnames].pct_change()  # Not selecting the benchmark for now
    returns = returns.drop(0)
    npreturns = returns.to_numpy()

    #  Storing the market caps & computing weights...
    mktcaps = market[mktcapsnames[1:]]
    mktcaps["SUM"] = mktcaps.sum(axis=1)
    weights = mktcaps.loc[:, mktcapsnames[1]:mktcapsnames[-1]].div(mktcaps["SUM"], axis=0)
    weights = weights.drop(0)
    npweight = weights.to_numpy()

    #Computing the prior

    expreturns, covars = assets_meanvar(np.transpose(npreturns))
    R = expreturns  # R is the vector of expected returns
    C = covars  # C is the covariance matrix
    # (rf is the risk free rate)
    W = npweight[-1]

    new_mean = compute_mean(W, R)
    new_var = compute_var(W, C)

    gamma = 3
    Pi = dot(dot(gamma, C), W)  # Compute equilibrium excess returns

    # Compute equilibrium excess returns taking into account views on assets
    sub_a = inv(dot(tau, C))
    sub_b = dot(dot(transpose(P), inv(omega)), P)
    sub_c = dot(inv(dot(tau, C)), Pi)
    sub_d = dot(dot(transpose(P), inv(omega)), Q)
    Pi_new = dot(inv(sub_a + sub_b), (sub_c + sub_d))

    # Perform a mean-variance optimization taking into account views
    WeightBL = 1 / lmbda * np.matmul(inv(C), Pi_new)
    return WeightBL

#%% try
P = np.array([[1, 0, 0, 0, 0, -1], [1, 0, 0, -1, 0, 0]])
# We say that the North American index will outperform the EM Asia Index...
Q = np.array([0.05, 0.05])  # By 5%
tau = 0.1  # tau is a scalar indicating the uncertainty
# There are two ways to define omega: first (not recommended)
# omega = dot(dot(dot(tau, P), C), transpose(P))

# Second, define it by hand
omega = np.array([[0.5, 0], [0, 0.3]])
lmbda = 2
rf = 0.001
print(weight(P, Q, tau, omega, lmbda, rf))