import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from matplotlib.ticker import FuncFormatter

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions, base_optimizer

from scipy.stats import norm
import math


def rx_mgt():
    st.subheader('Value @ Risk Analysis - Monte Carlo Simulations')

    # tickers = ['GOOGL', 'FB', 'AAPL', 'NFLX', 'AMZN']
    tickers = ["MSFT", "AMZN", "KO", "MA", "COST",
               "LUV", "XOM", "PFE", "JPM", "UNH",
               "ACN", "DIS", "GILD", "F", "TSLA"]
    n = len(tickers)
    price_data = []
    for ticker in range(n):
        prices = web.DataReader(tickers[ticker], start='2018-06-20', end='2020-06-20', data_source='yahoo')
        price_data.append(prices[['Adj Close']])
    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns = tickers
    st.dataframe(df_stocks.tail())

    weights = np.array([1 / len(tickers)] * len(tickers))

    # Annualized Return
    # mu = expected_returns.capm_return(df_stocks)
    mu = expected_returns.mean_historical_return(df_stocks)
    # Sample Variance of Portfolio
    # Sigma = risk_models.sample_cov(df_stocks)
    Sigma = risk_models.CovarianceShrinkage(df_stocks).ledoit_wolf()
    # Max Sharpe Ratio - Tangent to the EF
    initial_weights = weights

    # ef = EfficientFrontier(mu, Sigma)
    # You don't have to provide expected returns in this case
    ef = EfficientFrontier(mu, Sigma, weight_bounds=(None, None))
    # ef.add_objective(objective_functions.transaction_cost, w_prev=initial_weights, k=0.001)
    # ef.add_objective(objective_functions.L2_reg, gamma=0.08)  # default is 1
    ef.min_volatility()

    # weights = ef.clean_weights()
    # st.write(weights)
    # sharpe_pfolio = ef.max_sharpe()

    # May use add objective to ensure minimum zero weighting to individual stocks
    sharpe_pwt = ef.clean_weights()
    st.write('Weights for each asset: ', sharpe_pwt)

    # VaR Calculation
    ticker_rx2 = []
    # Convert Dictionary to list of asset weights from Max Sharpe Ratio Portfolio
    sh_wt = list(sharpe_pwt.values())
    sh_wt = np.array(sh_wt)

    for a in range(n):
        ticker_rx = df_stocks[[tickers[a]]].pct_change()
        ticker_rx = (ticker_rx + 1).cumprod()
        ticker_rx2.append(ticker_rx[[tickers[a]]])
    ticker_final = pd.concat(ticker_rx2, axis=1)

    st.write('Holding Period Returns: ', ticker_final)

    # Taking Latest Values of Return
    pret = []
    pre1 = []
    price = []
    for x in range(n):
        pret.append(ticker_final.iloc[[-1], [x]])
        price.append((df_stocks.iloc[[-1], [x]]))
    pre1 = pd.concat(pret, axis=1)
    pre1 = np.array(pre1)
    price = pd.concat(price, axis=1)
    varsigma = pre1.std()
    ex_rtn = pre1.dot(sh_wt)
    print('The weighted expected portfolio return for selected time period is' + str(ex_rtn))

    st.write('The weighted expected portfolio return for selected time period is' + str(ex_rtn))

    # ex_rtn = (ex_rtn)**0.5-(1) #Annualizing the cumulative return (will not affect outcome)
    price = price.dot(sh_wt)  # Calculating weighted value
    # print(ex_rtn, varsigma, price)
    # st.write(ex_rtn, varsigma, price)

    st.write('**Running Monte Carslo Simulations of 10000 runs 1440 minutes/day**')

    Time = 1440  # No of days(steps or trading days in this case)
    lt_price = []
    final_res = []
    for i in range(10000):  # 10000 runs of simulation
        daily_returns = (np.random.normal(ex_rtn / Time, varsigma / math.sqrt(Time), Time))

    print(np.percentile(daily_returns, 5),
          np.percentile(daily_returns,
                        95))  # VaR - Minimum loss of 5.7% at a 5% probability, also a gain can be higher than 15% with a 5 % probability
    min_loss = np.percentile(daily_returns * 100, 5)
    max_gain = np.percentile(daily_returns * 100, 95)

    st.write('Minimum loss @ 5% probability', round(min_loss, 2), '%')
    st.write('Gain @ 5% probability', round(max_gain, 2), '%')

    st.write('Assuming our portfolio value to be $10,000')

    pvalue = 10000  # portfolio value
    st.write('**$Amount required to cover minimum losses for one day is** ' +
             str(pvalue * - round(np.percentile(daily_returns, 5), 3)))


def write():
    rx_mgt()
    return
