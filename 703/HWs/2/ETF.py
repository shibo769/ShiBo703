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
        daily_return = (price_data - price_data.shift(1)) / price_data.shift(1)
        daily_return = daily_return.dropna()
        cumulative_return = (price_data.iloc[-1,:] - price_data.iloc[0,:]) / price_data.iloc[0,:]
        hold_days = len(price_data)
        annualized_return = (1 + cumulative_return) ** (365/hold_days) - 1
        return annualized_return
    
    def annualized_std(self):
        daily_return = (price_data - price_data.shift(1)) / price_data.shift(1)
        daily_return = daily_return.dropna()
        annualized_std = np.std(daily_return) * math.sqrt(252) 
        return annualized_std
    
    def return_data1(self,freq=None):
        if freq is None:
            return_data = np.log(self.price_data.pct_change().dropna() + 1)
        else:
            return_data = np.log(self.price_data.resample(freq).ffill().pct_change().dropna() + 1)
        self.return_data = return_data
        return self.return_data
  
    def matrices(self, data_type, matrix_type):
        if matrix_type == 'cov':
            return data_type.cov()
        if matrix_type == 'corr':
            return data_type.corr()
    