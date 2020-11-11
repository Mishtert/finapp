import streamlit as st
from pandas_datareader import data as web
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from pypfopt.efficient_frontier import EfficientFrontier, objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns

# assets = ["MSFT", "AMZN", "ACN", "MA", "TSLA",]
# assets = ["BLK", "BAC", "AAPL", "TM", "WMT",
#           "JD", "INTU", "MA", "UL", "CVS",
#           "DIS", "AMD", "NVDA", "PBI", "TGT"]

assets = ["MSFT", "AMZN", "KO", "MA", "COST",
          "LUV", "XOM", "PFE", "JPM", "UNH",
          "ACN", "DIS", "GILD", "F", "TSLA"]


# assets = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]

def optimize():
    st.markdown('## Portfolio Reallocation & Optimization with Cost Objective')

    weights = np.array([1 / len(assets)] * len(assets))
    # Create a dataframe to store the adjusted close price of the stocks
    df = pd.DataFrame()

    ohlc = yf.download(assets, period="max")
    prices = ohlc["Adj Close"]
    df = prices[prices.index <= "2020-08-31"]
    # Print the name of stocks
    st.write('Stocks under consideration are:')

    rslt =[]
    for i in df.columns[0:]:
        rslt.append(i)
    st.write(rslt)

    # st.dataframe(df.tail())

    # Create the title 'Portfolio Adj Close Price History
    title = 'Portfolio Adj. Close Price History    '

    # Get the stocks
    my_stocks = df[df.index >= "2008-01-01"]
    # Create and plot the graph
    plt.figure(figsize=(12.2, 4.5))  # width = 12.2in, height = 4.5
    # Loop through each stock and plot the Adj Close for each day
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label=c)  # plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj. Price USD ($)', fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    # plt.show()
    st.pyplot(plt)

    st.markdown('### Simple Returns based on equal allocation for all assets')

    st.write(weights)
    # Show the daily simple returns, NOTE: Formula = new_price/old_price - 1
    returns = df.pct_change()

    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)
    portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252

    percent_var = str(round(port_variance * 100, 2)) + '%'
    percent_vols = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

    st.markdown("##### Expected annual return : " + percent_ret)
    st.markdown("##### Annual volatility/standard deviation/risk : " + percent_vols)
    st.markdown("##### Annual variance : " + percent_var)

    st.markdown('#### annual volatility/risk is high')

    st.markdown('## Optimizing the mean-variance')

    mu = expected_returns.capm_return(df)  # returns.mean() * 252
    S = risk_models.semicovariance(df)  # Get the sample covariance matrix

    st.markdown("### Min volatility with a transaction cost objective")

    st.write(
        'Let us say that you already have a portfolio, and want to now optimise it. '
        'It could be quite expensive to completely reallocate, so you may want to take into account transaction'
        'costs')
    initial_weights = weights

    ef = EfficientFrontier(mu, S)

    st.write('1% broker commission')
    ef.add_objective(objective_functions.transaction_cost, w_prev=initial_weights, k=0.01)
    ef.min_volatility()
    weights = ef.clean_weights()
    st.write(weights)

    # st.markdown('##### we notice that one stock have now been allocated zero weight, let us fix that')
    #
    # ef = EfficientFrontier(mu, S)
    # ef.add_objective(objective_functions.transaction_cost, w_prev=initial_weights, k=0.001)
    # ef.add_objective(objective_functions.L2_reg)
    # ef.min_volatility()
    # weights = ef.clean_weights()
    # st.write(weights)

    st.markdown('##### Performing re-allocation since they are of equal weights')

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.transaction_cost, w_prev=initial_weights, k=0.001)
    ef.add_objective(objective_functions.L2_reg, gamma=0.08)  # default is 1
    ef.min_volatility()
    weights = ef.clean_weights()
    st.write(weights)
    output = ef.portfolio_performance();
    # st.write(output)
    annual_return = str(round(output[0] * 100, 2)) + '%'
    annual_volatility = str(round(output[1] * 100, 2)) + '%'
    sharpe_ratio = str(round(output[2], 2))

    st.markdown('##### Expected annual return : ' + annual_return)
    st.markdown('##### Annual volatility/standard deviation/risk : ' + annual_volatility)
    st.markdown('##### Sharpe Ratio : ' + sharpe_ratio)

    st.markdown('### Assuming I am willing to invest $20000, '
                'how should the allocation be')

    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    latest_prices = get_latest_prices(df)
    weights = weights
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=20000)
    allocation, leftover = da.lp_portfolio()

    st.write((f"Discrete allocation performed with ${leftover:.2f} leftover"))
    for key, value in allocation.items():
        st.write(key, value)


def write():
    optimize()
    return
