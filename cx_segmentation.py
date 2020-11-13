import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

# url = "https://github.com/Mishtert/bankcxsegmentation/blob/main/CC%20GENERAL.csv?raw=true"
# creditcard_df = pd.read_csv(url, index_col=0)
# st.dataframe(creditcard_df.head(2))
# creditcard_df = creditcard_df.reset_index()
# st.dataframe(creditcard_df.head(2))


def segmentation():
    st.subheader('Customer Segmentation(Credit Card Transactions)')
    st.markdown('### About data')
    
    url = "https://github.com/Mishtert/bankcxsegmentation/blob/main/CC%20GENERAL.csv?raw=true"
    creditcard_df = pd.read_csv(url, index_col=0)
#     st.dataframe(creditcard_df.head(2))
    creditcard_df = creditcard_df.reset_index()
    st.dataframe(creditcard_df.head(2))

#     st.dataframe(creditcard_df.head())
    st.write('1. **CUSTID:** Identification of Credit Card holder \n')
    st.write('2. **BALANCE:** Balance amount left in customers account to make purchases \n')
    st.write(
        '3. **BALANCE_FREQUENCY:** How frequently the Balance is updated,score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)\n')
    st.write('4. **PURCHASES:** Amount of purchases made from account\n')
    st.write('5. **ONEOFFPURCHASES:** Maximum purchase amount done in one-go\n')
    st.write('6. **INSTALLMENTS_PURCHASES:** Amount of purchase done in installment\n')
    st.write('7. **CASH_ADVANCE:** Cash in advance given by the user\n')
    st.write(
        '8. **PURCHASES_FREQUENCY:** How frequently the Purchases are being made,score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)\n')
    st.write(
        '9. **ONEOFF_PURCHASES_FREQUENCY:** How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)\n')
    st.write(
        '10. **PURCHASES_INSTALLMENTS_FREQUENCY:** How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)\n')
    st.write('11. **CASH_ADVANCE_FREQUENCY:** How frequently the cash in advance being paid\n')
    st.write('12. **CASH_ADVANCE_TRX:** Number of Transactions made with "Cash in Advance"\n')
    st.write('13. **PURCHASES_TRX:** Number of purchase transactions made\n')
    st.write('14. **CREDIT_LIMIT:** Limit of Credit Card for user\n')
    st.write('15. **PAYMENTS:** Amount of Payment done by user\n')
    st.write('16. **MINIMUM_PAYMENTS:** Minimum amount of payments made by user\n')
    st.write('17. **PRC_FULL_PAYMENT:** Percent of full payment paid by user\n')
    st.write('18. **TENURE:** Tenure of credit card service for user\n')
    st.write('\n')
    st.write('There are 18 features with 8950 observations\n')
    st.markdown('### Some High-level Information of the data available\n')
    st.markdown('1. Mean balance is $1564\n')
    st.markdown('2. Balance frequency is frequently updated on average ~0.9\n')
    st.markdown('3. Purchases average is $1000\n')
    st.markdown('4. one off purchase average is ~$600\n')
    st.markdown('5. Average purchases frequency is around 0.5\n')
    st.markdown(
        '6. Average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low\n')
    st.markdown('7. Average credit limit ~ 4500\n')
    st.markdown('8. Percent of full payment is 15%\n')
    st.markdown('9. Average tenure is 11 years\n')

    st.markdown('#### Customer who had higher one-off purchase\n')

    cx_oop = creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == creditcard_df['ONEOFF_PURCHASES'].max()]
    st.dataframe(cx_oop)

    st.markdown('#### Customer with high cash advance')
    cx_ca = creditcard_df[creditcard_df['CASH_ADVANCE'] == creditcard_df['CASH_ADVANCE'].max()]
    st.dataframe(cx_ca)
    st.markdown('##### This customer made 123 cash advance transactions!! # Never paid credit card in full ')

    # Fill up the missing elements with mean of the 'MINIMUM_PAYMENT'
    creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df[
        'MINIMUM_PAYMENTS'].mean()
    # Fill up the missing elements with mean of the 'CREDIT_LIMIT'
    creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df[
        'CREDIT_LIMIT'].mean()
    # Let's drop Customer ID since it has no meaning here
    creditcard_df.drop(["CUST_ID","Recnum"], axis=1, inplace=True)

    st.markdown('### Let us now group customers based on some common factors')
    # Let's scale the data first
    scaler = StandardScaler()
    creditcard_df_scaled = scaler.fit_transform(creditcard_df)
    # Apply K-means
    kmeans = KMeans(8)
    kmeans.fit(creditcard_df_scaled)
    labels = kmeans.labels_
    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=[creditcard_df.columns])

    # In order to understand what these numbers mean, let's perform inverse transformation
    cluster_centers = scaler.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(data=cluster_centers, columns=[creditcard_df.columns])
    st.dataframe(cluster_centers)

    st.write('**Some observation on clustered customers**')
    st.write('1. **First Customers cluster (Transactors):** Those are customers who pay least amount of  '
             'charges and careful with their money, Cluster with lowest balance ($104) and '
             'cash advance ($303), Percentage of full payment = 23%')
    st.write('2. **Second customers cluster (revolvers):** Who use credit card as a loan '
             '(most lucrative sector): highest balance ($5000) and cash advance (~$5000), '
             'low purchase frequency, high cash advance frequency (0.5), '
             'high cash advance transactions (16) and low percentage of full payment (3%)')
    st.write('3. **Third customer cluster (VIP/Prime):** High credit limit $16K and '
             'highest percentage of full payment, target for increase credit limit and increase spending habits')
    st.write('4. **Fourth customer cluster (low tenure):** These are customers with low tenure (7 years), low balance ')

    # y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
    # concatenate the clusters labels to our original dataframe
    creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis=1)
    st.dataframe(creditcard_df_cluster.head())
    st.markdown('### We can group any new customers on one of these clusters and create marketing '
                'campaign specific to these clusters')


def write():
    segmentation()
    return
