import streamlit as st
import numpy as np
import pandas as pd
import sys
import math
from collections import defaultdict
import matplotlib.pyplot as plt



url = 'https://github.com/Mishtert/cardtransactiondata/blob/main/card%20transactions.csv?raw=true'
# data = pd.read_csv(url, index_col=0)
data = pd.read_csv(url, index_col=False)
df = data.reset_index()
df = df.dropna(axis=1, how='all')
df['first_digit'] = df['Amount'].multiply(100).apply(lambda row: str(row)[0])

# filter for only transaction type P
df = df[df.Transtype == 'P']

# Removing Fedex
df = df[(df['Merch description'].str.lower().str.contains('fedex') == False)]

print('null values:', df.Amount.isnull().sum())

# Benford's Law percentages for leading digits 1-9
BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]


def count_first_digit(data_str):  # TAKE AS AN ARGUMENT A STR-COLUMN NAME
    mask = df[data_str] > 1.
    data = list(df[mask][data_str])
    for i in range(len(data)):
        while data[i] > 10:
            data[i] = data[i] / 10
    first_digits = [int(x) for x in sorted(data)]
    unique = (set(first_digits))  # a list with unique values of first_digit list
    data_count = []
    for i in unique:
        count = first_digits.count(i)
        data_count.append(count)
    total_count = sum(data_count)
    data_percentage = [(i / total_count) * 100 for i in data_count]
    return total_count, data_count, data_percentage


def get_expected_counts(total_count):
    """Return list of expected Benford's Law counts for total sample count."""

    return [round(p * total_count / 100) for p in BENFORD]


# expected_counts = get_expected_counts(total_count)

def chi_square_test(data_count, expected_counts):
    """Return boolean on chi-square test (8 degrees of freedom & P-val=0.05)."""

    chi_square_stat = 0  # chi square test statistic

    for data, expected in zip(data_count, expected_counts):
        chi_square = math.pow(data - expected, 2)

        chi_square_stat += chi_square / expected

    print("\nChi-squared Test Statistic = {:.3f}".format(chi_square_stat))
    st.write("\nChi-squared Test Statistic = {:.3f}".format(chi_square_stat))

    print("Critical value at a P-value of 0.05 is 15.51.")
    st.write("Critical value at a P-value of 0.05 is 15.51.")

    return chi_square_stat < 15.51


# chi_square_test(data_count, expected_counts)

# 1st_bar_chart
def bar_chart(data_pct):
    """Make bar chart of observed vs expected 1st digit frequency in percent."""

    fig, ax = plt.subplots()

    index = [i + 1 for i in range(len(data_pct))]  # 1st digits for x-axis
    # text for labels, title and ticks

    fig.canvas.set_window_title('Percentage First Digits')
    ax.set_title('Data vs. Benford Values', fontsize=15)
    ax.set_ylabel('Frequency (%)', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(index, fontsize=14)

    # build bars
    rects = ax.bar(index, data_pct, width=0.95, color='black', label='Data')
    # attach a text label above each bar displaying its height

    for rect in rects:
        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width() / 2, height,

                '{:0.1f}'.format(height), ha='center', va='bottom',

                fontsize=13)

        # plot Benford values as red dots
        ax.scatter(index, BENFORD, s=150, c='red', zorder=2, label='Benford')
        # Hide the right and top spines & add legend
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(prop={'size': 15}, frameon=False)
        plt.show()
        st.pyplot(fig, ax)

        # 2nd_bar_chart
        labels = list(data_percentage)
        width = 0.35
        x = np.arange(len(data_percentage))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, data_percentage, width=0.95, color='black', label='Data')
        rects2 = ax.bar(x + width, BENFORD, width, label='Benford')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Frequency (%)', fontsize=16)
        ax.set_title('Benford')
        ax.set_xticks(x)
        ax.legend()
        plt.show()
        st.pyplot(fig, ax)


# specify the main() function and runs the program & prints some statistics.

def main(data_list):
    st.subheader('Potential Fraud/Anomaly Transaction Identification')

    st.write('### About Data')

    st.write('Credit Card Transactions in 2010 from governmental organizations. '
             'Cleaned for academic purpose of building a supervised fraud algorithm. '
             'Dataset has 96,753 records and 10 fields.')
    st.write('we only consider "P" transactions and exclude transactions from "FedEx". '
             'There’re 84,623 records')

    st.write("**Benford's law has been used in two different ways here"
             "(expected vs. observed spread of digits from 1-9 & "
             "low-digit numbers (1,2) account for about 47.7%)**")
    st.write('#### Sample of Dataset')
    st.dataframe(df.head(5))
    total_count, data_count, data_percentage = count_first_digit(data_list)
    expected_counts = get_expected_counts(total_count)

    print("\nobserved counts = {}".format(data_count))
    # st.write("\nobserved counts = {}".format(data_count))

    print("expected counts = {}".format(expected_counts), "\n")
    # st.write("expected counts = {}".format(expected_counts), "\n")

    print("First Digit Probabilities:")
    st.write("**First Digit Probabilities:**")

    for i in range(1, 10):
        actual_digit = (data_percentage[i - 1] / 100) * 100
        ben_digit = (BENFORD[i - 1] / 100) * 100
        msg = "{}: observed: {:.2f}  expected: {:.2f}".format(i, actual_digit, ben_digit)
        # print("{}: observed: {:.3f}  expected: {:.3f}")
        print(msg)
        # st.write("{}: observed: {:.3f}  expected: {:.3f}")
        st.write(msg)
        # format(i, data_percentage[i - 1] / 100, BENFORD[i - 1] / 100)
        # format(i, actual_digit, ben_digit)

    if chi_square_test(data_count, expected_counts):

        print("Observed distribution matches expected distribution.")
        st.write("**Observed distribution matches expected distribution**.")

    else:
        print("Observed distribution does not match expected.", file=sys.stderr)
        st.write("**Observed distribution does not match expected**.", file=sys.stderr)

    # bar_chart(data_percentage)


def benfordstat(series, n_mid=15, c=3):
    count_low = sum(pd.to_numeric(series) <= 2)
    count_all = len(series)
    r = count_low / count_all / 0.477
    t = (count_all - n_mid) / c
    new_r = 1 + (r - 1) / (1 + np.exp(-t))
    stat = max(new_r, 1 / new_r)
    return (stat)


def analyze():
    st.write('##### Higher the BenfordStat, more the scrutiny required. Ideal score should be close to 1')
    # MERCHNUM
    MN_stat = df.groupby(['Merchnum'])['first_digit'].apply(benfordstat).reset_index()
    MN_stat.columns = ['Merchnum', 'BenFordStat']
    st.write(MN_stat.head(10))
    # CARDNUM
    CN_stat = df.groupby(['Cardnum'])['first_digit'].apply(benfordstat).reset_index()
    CN_stat.columns = ['Cardnum', 'BenFordStat']
    st.write(CN_stat.head(10))
    st.write('#### Sorting values by the unusualness scores and get 10 records with the highest scores')
    st.write(CN_stat.sort_values(by='BenFordStat', ascending=False).head(10))
    st.write(MN_stat.sort_values(by='BenFordStat', ascending=False).head(10))

    st.write('One Merchant Number **991808369338** has has high unusual score')
    st.dataframe(df[df['Merchnum'] == '991808369338'][['Recnum', 'Cardnum', 'Date', 'Amount', 'first_digit']])
    st.write('The merchandizer charged the same amount of money and charged only one card number. '
             'The transactions occurred several times a day in some days. This warrants closer investigation'
             'This could be a genuine transaction too.')


def write():
    # detect()

    main('Amount')
    analyze()

    return
