# -*- coding: utf-8 -*-
"""
Assignment 1 for MF 703: Programming for Mathematical Finance
Professor: Chris Kelliher
Author: SHI BO
Email: shibo@bu.edu
Date: 09-07-2020
"""
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

    
class ETF():
    def __init__(self, ticker):
        self.ticker = ticker
        
    def get_data(self, start_data, end_data):
        data = yf.download(self.ticker,start=start_data,end=end_data)
        self.price_data = data['Adj Close']
        return self.price_data
    
    def annualized_return(self):
        return_annualized = (price_data.iloc[-1, :] / price_data.iloc[0, :]) ** (252 / len(price_data)) - 1
        return return_annualized
    
    def annualized_std(self):
        return_annualized = (price_data.iloc[-1, :] / price_data.iloc[0, :]) ** (252 / len(price_data)) - 1
        std_annualized = return_annualized.std()
        return std_annualized
    
    def return_data1(self,freq=None):
        if freq is None:
            return_data = np.log(self.price_data.pct_change().dropna() + 1)
        else:
            return_data = np.log(self.price_data.resample(freq).ffill().pct_change().dropna() + 1)
        self.return_data = return_data
        return self.return_data
  
def matrices(data_type, matrix_type):
      if matrix_type == 'cov':
        return data_type.cov()
      elif matrix_type == 'corr':
        return data_type.corr()

def rolling_corr(data_type, window, column):
    rolling_corr = data_type.rolling(window).corr(data_type[column])
    rolling_corr.plot(figsize=(12, 8))
    plt.xlabel('Date',fontsize = 15)
    plt.ylabel('Correlation',fontsize = 15)
    plt.title( '90-day correlation of each sector ETF with the S&P index',fontsize = 20)
    plt.show()
    return rolling_corr.dropna()

def Linear_reg(X, y,value = 'nor'):
    df = pd.concat([X, y], axis=1).dropna()
    mod = sm.OLS(df.iloc[:, 1], df.iloc[:, 0])
    result = mod.fit()
    if value == 'nor':
        return result.params[0]

def CAPM(data_type, column):
    Y = data_type[column]
    result = data_type[tickers].apply(lambda x: Linear_reg(x, Y))
    return pd.DataFrame(result, columns=['beta'])

def rolling_CAPM(data_type, window, column):
    X = data_type[column]
    rolling_beta = data_type[tickers].rolling(window).apply(lambda y: Linear_reg(X, y)).dropna()
    rolling_beta.plot(figsize=(12, 8))
    plt.ylabel('beta', fontsize = 15 )
    plt.xlabel('date', fontsize = 15 )
    plt.title('Rolling Beta 90-days', fontsize = 25 )
    plt.show()
    return rolling_beta

def auto(data_type):
    ar_coeff = data_type.apply(lambda x: Linear_reg(x, x.shift(1)))
    return ar_coeff
    
if __name__ == '__main__':
    # 1.download
    tickers = ['SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
    ETFs = ETF(tickers)
    print('1) Download price data of ETFs from January 1st 2010:')
    price_data = ETFs.get_data("2010-01-01", "2020-09-06")
    return_data = ETFs.return_data1()
    check_ano = pd.concat([return_data.isna().sum(), price_data.isna().sum()], axis=1)
    check_ano.columns = ['price_data_ano','return_data_ano']
    print('check whehter there are anomolies: \n',check_ano)
    # 2.Annualized Return and Standard Deviation
    r, s = ETFs.annualized_return(),ETFs.annualized_std()
    return_std_df = pd.concat([r, s], axis=1)
    return_std_df.columns = ['Ann_Return', 'Standard_Deviation']
    print('2) Annualized Return and Standard Deviation :')
    print(return_std_df.T)
    # 3.covaiance matrix
    return_data_month = ETFs.return_data1(freq='M')
    corr_matrix_day = matrices(return_data, 'corr')
    corr_matrix_month = matrices(return_data_month, 'corr')
    cov_matrix_day = matrices(return_data, 'cov')
    cov_matrix_month = matrices(return_data_month, 'cov')
    diff_matrix = corr_matrix_day - corr_matrix_month
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.5)
    aa = sns.heatmap(corr_matrix_day, annot = True)
    aa.set_title('correlation matrix for daily return',fontsize=35)
    plt.show()
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.5)
    bb = sns.heatmap(corr_matrix_month, annot = True)
    bb.set_title('correlation matrix for monthly return',fontsize=35)
    plt.show()
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.5)
    cc = sns.heatmap(cov_matrix_day, annot = True)
    cc.set_title('covariance matrix for daily return',fontsize=35)
    plt.show()
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.5)
    dd = sns.heatmap(cov_matrix_month, annot = True)
    dd.set_title('covariance matrix for monthly return',fontsize=35) 
    plt.show()
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.5)
    ee = sns.heatmap(diff_matrix, annot = True)
    ee.set_title('correlation matrix for diff_matrix',fontsize=35)
    plt.show()
    # 4. 90-days corr
    print('4) Rolling Correlation with SPY:')
    rolling_corr = rolling_corr(return_data, 90, column='SPY')
    print(rolling_corr)
    # 5.CAPM
    print('5) CAPM model:')
    beta = CAPM(return_data, 'SPY')
    rolling_beta = rolling_CAPM(return_data, 90, 'SPY')
    print(beta)
    # 6.auto-regression
    ar_coeff = auto(return_data)
    print('6) AutoRegression Coefficient:')
    print(ar_coeff.T)
    
    

    


    
