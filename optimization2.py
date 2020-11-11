import streamlit as st
from pandas_datareader import data as web
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from pypfopt.efficient_frontier import EfficientFrontier, objective_functions
from pypfopt import risk_models, plotting, expected_returns, DiscreteAllocation, get_latest_prices

assets = ["MSFT", "AMZN", "KO", "MA", "COST",
          "LUV", "XOM", "PFE", "JPM", "UNH",
          "ACN", "DIS", "GILD", "F", "TSLA"]


def optimize2():
    st.header('Portfolio Optimization - (Optimizing Risk-Return/Minimizing Risk/Fixed Return)')

    weights = np.array([1 / len(assets)] * len(assets))
    # Create a dataframe to store the adjusted close price of the stocks
    df = pd.DataFrame()

    ohlc = yf.download(assets, period="max")
    prices = ohlc["Adj Close"]
    df = prices[prices.index <= "2020-08-31"]

    # Print the name of stocks
    st.write('Stocks under consideration are:')

    rslt = []
    for i in df.columns[0:]:
        rslt.append(i)
    st.write(rslt)

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

    # st.markdown('##### Calculating Covariance Matrix')
    sample_cov = risk_models.sample_cov(prices, frequency=252)
    # st.write(sample_cov)

    st.markdown('##### Return estimation')
    mu = expected_returns.capm_return(prices)
    st.write(mu)
    # muplot = mu.plot.barh(figsize=(10,6))
    # st.pyplot(muplot)

    st.markdown('##### Long / short min variance')
    st.write('let us construct a long/short portfolio with the objective of minimising variance')
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    # You don't have to provide expected returns in this case
    ef = EfficientFrontier(None, S, weight_bounds=(None, None))
    ef.min_volatility()
    weights = ef.clean_weights()
    st.write(weights)

    st.markdown('Let us get a quick indication of the portfolio performance '
                '(this is an in sample estimate and may have very little resemblance to how the portfolio actually '
                'performs!)')
    output = ef.portfolio_performance(verbose=True);
    annual_volatility = str(round(output[1] * 100, 2)) + '%'

    st.markdown('##### Annual volatility/standard deviation/risk : ' + annual_volatility)

    st.markdown('### Assuming I am willing to invest $20000 and would like to have a 130/30 long/short portfolio,'
                'the allocation will be')

    latest_prices = get_latest_prices(df)  # prices as of the day you are allocating
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=20000, short_ratio=0.3)
    allocation, leftover = da.lp_portfolio()
    print(f"Discrete allocation performed with ${leftover:.2f} leftover")

    st.write(f"Discrete allocation performed with ${leftover:.2f} leftover")
    for key, value in allocation.items():
        st.write(key, value)

    st.markdown('### Theoretically maximising the Sharpe ratio gives the optimal portfolio in terms of risks-returns.'
                'Let us build a long-only max-sharpe portfolio with sector constraints')

    sector_mapper = {
        "MSFT": "Tech",
        "AMZN": "Consumer Discretionary",
        "KO": "Consumer Staples",
        "MA": "Financial Services",
        "COST": "Consumer Staples",
        "LUV": "Aerospace",
        "XOM": "Energy",
        "PFE": "Healthcare",
        "JPM": "Financial Services",
        "UNH": "Healthcare",
        "ACN": "Misc",
        "DIS": "Media",
        "GILD": "Healthcare",
        "F": "Auto",
        "TSLA": "Auto"
    }

    sector_lower = {
        "Consumer Staples": 0.1,  # at least 10% to staples
        "Tech": 0.05  # at least 5% to tech
        # For all other sectors, it will be assumed there is no lower bound
    }

    sector_upper = {
        "Tech": 0.2,
        "Aerospace": 0.1,
        "Energy": 0.1,
        "Auto": 0.15
    }

    st.markdown('##### some of the constraints (lower):')
    st.markdown('* at least 10% to Staples')
    st.markdown('* at least 5% to Technology')
    st.markdown('* For other sectors, we are assuming that there will be no lower bound')

    st.markdown('##### some of the constraints (upper):')
    st.markdown('* maximum of 20% to Technology')
    st.markdown('* maximum of 10% to Aerospace')
    st.markdown('* maximum of 10% to Energy')
    st.markdown('* maximum of 15% to Auto')
    st.write('In addition to above I have two more specific constraints')
    st.markdown('1. 10% of portfolio should be in AMZN')
    st.markdown('2. Less than 5% of my portfolio in TSLA')

    mu = expected_returns.capm_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)  # weight_bounds automatically set to (0, 1)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    amzn_index = ef.tickers.index("AMZN")
    ef.add_constraint(lambda w: w[amzn_index] == 0.10)

    tsla_index = ef.tickers.index("TSLA")
    ef.add_constraint(lambda w: w[tsla_index] <= 0.05)

    ef.add_constraint(lambda w: w[10] >= 0.05)

    ef.max_sharpe()
    weights = ef.clean_weights()

    st.write(weights)

    st.markdown('##### Checking if constraints were met')
    for sector in set(sector_mapper.values()):
        total_weight = 0
        for t, w in weights.items():
            if sector_mapper[t] == sector:
                total_weight += w
        st.write(f"{sector}: {total_weight:.3f}")

    st.markdown('## Let us now maximise the returns for a given risk')
    st.write('let us assume that our upper limit for volatility is 15% and we cannot accept anything more than that')
    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    ef.efficient_risk(target_volatility=0.15)
    weights = ef.clean_weights()
    st.write(weights)

    num_small = len([k for k in weights if weights[k] <= 1e-4])
    st.write(f"{num_small}/{len(ef.tickers)} assets have zero weight")

    output = ef.portfolio_performance();
    # st.write(output)
    annual_return = str(round(output[0] * 100, 2)) + '%'
    annual_volatility = str(round(output[1] * 100, 2)) + '%'
    sharpe_ratio = str(round(output[2], 2))

    st.markdown('##### Expected annual return : ' + annual_return)
    st.markdown('##### Annual volatility/standard deviation/risk : ' + annual_volatility)
    st.markdown('##### Sharpe Ratio : ' + sharpe_ratio)

    st.write('Although it seems that our objectives are met, we may get better results by adding '
             'some level of diversification (Some assets have zero allocation)')
    # You must always create a new efficient frontier object
    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamme is the tuning parameter
    ef.efficient_risk(0.15)
    weights = ef.clean_weights()
    st.write(weights)

    st.write('Let us check how many assets are with zero weights')
    num_small = len([k for k in weights if weights[k] <= 1e-4])
    st.write(f"{num_small}/{len(ef.tickers)} assets have zero weight")

    st.write('We can fine tune the allocation spread')

    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    ef.add_objective(objective_functions.L2_reg, gamma=1)  # gamme is the tuning parameter
    ef.efficient_risk(0.15)
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

    st.markdown('## Minimise risk for a given return, market-neutral')
    st.write('what if we require a certain return rate (8%) and our portfolio to be market neutral?')

    # Must have no weight bounds to allow shorts
    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_objective(objective_functions.L2_reg)
    ef.efficient_return(target_return=0.08, market_neutral=True)
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

    st.markdown('## Unconstrained Efficient Frontier')

    from pypfopt import CLA, plotting

    cla = CLA(mu, S)
    cla.max_sharpe()
    weights = cla.clean_weights()
    output = cla.portfolio_performance(verbose=True);

    st.write(weights)
    annual_return = str(round(output[0] * 100, 2)) + '%'
    annual_volatility = str(round(output[1] * 100, 2)) + '%'
    sharpe_ratio = str(round(output[2], 2))

    st.markdown('##### Expected annual return : ' + annual_return)
    st.markdown('##### Annual volatility/standard deviation/risk : ' + annual_volatility)
    st.markdown('##### Sharpe Ratio : ' + sharpe_ratio)


def write():
    optimize2()
    return
