import streamlit as st
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import requests
import datetime
import time
import numpy as np

yf.pdr_override()
np.random.seed(1234)


def stock_screener():
    st.subheader('Screening Stocks (S&P500)')
    st.subheader('A sample page to screen s&p 500 stocks with multiple conditions')
    st.write('##### page loading will be slow since 500 stocks are analyzed for multiple conditions')
    st.markdown('### Some of the conditions to shortlist')
    st.markdown('1.  current price > 150 & 200 day simple moving averages (SMA)')
    st.markdown('2.  current price > 50-day SMA')
    st.markdown('3.  current price at least 30% > than 52-week low')
    st.markdown('4.  current price must be within 25% of 52-week high')
    st.markdown('5.  150-days SMA > 200-days SMA')
    st.markdown('6.  200-days SMA must be trending at least for 30 days')
    st.markdown('7.  50-days SMA must be > 150&200-days SMA')
    st.markdown('8.  IBD RS-Rating >=70')

    stock_list = si.tickers_sp500()
    # stock_list = si.tickers_nasdaq()
    index_name = '^GSPC'  # S&P 500
    final = []
    index = []
    n = -1
    stock_names = []
    export_list = pd.DataFrame(
        columns=['Stock', "rs_rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])

    for stock in stock_list[0:170]:
        n += 1
        time.sleep(1)

        print("\npulling {} with index {}".format(stock, n))

        # rs_rating
        # start_date = datetime.datetime.today() - datetime.timedelta(days=785)
        # end_date = datetime.date.today() - datetime.timedelta(days=365)
        # print(start_date)

        start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        end_date = datetime.date.today()
        # end_date = '2019-11-08'
        print(end_date)
        df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
        df['Percent Change'] = df['Adj Close'].pct_change()
        stock_return = df['Percent Change'].sum() * 100

        index_df = pdr.get_data_yahoo(index_name, start=start_date, end=end_date)
        index_df['Percent Change'] = index_df['Adj Close'].pct_change()
        index_return = index_df['Percent Change'].sum() * 100

        rs_rating = round((stock_return / index_return) * 10, 2)

        try:
            sma = [50, 150, 200]
            for x in sma:
                df["SMA_" + str(x)] = round(df.iloc[:, 4].rolling(window=x).mean(), 2)

            current_close = df["Adj Close"][-1]
            moving_average_50 = df["SMA_50"][-1]
            moving_average_150 = df["SMA_150"][-1]
            moving_average_200 = df["SMA_200"][-1]
            low_of_52week = min(df["Adj Close"][-260:])
            high_of_52week = max(df["Adj Close"][-260:])

            try:
                moving_average_200_20 = df["SMA_200"][-20]

            except Exception:
                moving_average_200_20 = 0

            # Condition 1: Current Price > 150 SMA and > 200 SMA
            if current_close > moving_average_150 > moving_average_200:
                condition_1 = True
            else:
                condition_1 = False
            # Condition 2: 150 SMA and > 200 SMA
            if moving_average_150 > moving_average_200:
                condition_2 = True
            else:
                condition_2 = False
            # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
            if moving_average_200 > moving_average_200_20:
                condition_3 = True
            else:
                condition_3 = False
            # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
            if moving_average_50 > moving_average_150 > moving_average_200:
                # print("Condition 4 met")
                condition_4 = True
            else:
                # print("Condition 4 not met")
                condition_4 = False
            # Condition 5: Current Price > 50 SMA
            if current_close > moving_average_50:
                condition_5 = True
            else:
                condition_5 = False
            # Condition 6: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before
            # coming out of consolidation)
            if current_close >= (1.3 * low_of_52week):
                condition_6 = True
            else:
                condition_6 = False
            # Condition 7: Current Price is within 25% of 52 week high
            if current_close >= (.75 * high_of_52week):
                condition_7 = True
            else:
                condition_7 = False

            # Condition 8: IBD rs_rating greater than 70
            if rs_rating >= 70:
                condition_8 = True
            else:
                condition_8 = False

            if (
                    condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7 and condition_8):
                final.append(stock)
                index.append(n)

                dataframe = pd.DataFrame(list(zip(final, index)), columns=['Company', 'Index'])

                # dataframe.to_csv('stocks.csv')

                export_list = export_list.append(
                    {'Stock': stock, "rs_rating": rs_rating, "50 Day MA": moving_average_50,
                     "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200,
                     "52 Week Low": low_of_52week, "52 week High": high_of_52week},
                    ignore_index=True)
                st.markdown(stock + " made the requirements")
        except Exception as e:
            print(e)
            print("No data on " + stock)

    st.dataframe(export_list)
    st.markdown('### Stats on the short-listed items')
    # st.write(export_list.iloc[:, 0])
    stats_list = list(export_list.iloc[:, 0])
    for tickers in stats_list:
        st.write(tickers)
        st.markdown('#### Abbreviation Guide:')
        st.markdown('###### mrq = Most Recent Quarter')
        st.markdown('###### ttm = Trailing Twelve Months')
        st.markdown('###### yoy = Year Over Year')
        st.markdown('###### lfy = Last Fiscal Year')
        st.markdown('###### fye = Fiscal Year Ending')
        st.markdown('##### Footnotes')
        st.markdown('###### 1 Data provided by Thomson Reuters.')
        st.markdown('###### 2 Data provided by EDGAR Online.')
        st.markdown('###### 3 Data derived from multiple sources or calculated by Yahoo Finance.')
        st.markdown('###### 4 Data provided by Morningstar, Inc.')
        st.markdown('###### 5 Shares outstanding is taken from the most recently filed quarterly or '
                    'annual report and Market Cap is calculated using shares outstanding.')
        st.markdown('###### 6 EBITDA is calculated by Capital IQ using methodology that may differ '
                    'from that used by a company in its reporting.')

        st.dataframe(si.get_stats(tickers))
        # st.write(stock_names)


def write():
    stock_screener()
    return
