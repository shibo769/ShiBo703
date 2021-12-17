import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa import stattools
from ETF import *
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import math
from statsmodels.tsa.stattools import adfuller

def BSM_put(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = stats.norm.cdf(-d1)
    Nd2 = stats.norm.cdf(-d2)
    put_price = Nd2 * K * np.exp(-r * t) - S * Nd1
    return put_price

def BSM_call(S, K, sigma, t, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    d2 = (np.log(S / K) + (r - sigma ** 2 / 2) * t) / (sigma * t ** 0.5)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    call_price = -Nd2 * K * np.exp(-r * t) + S * Nd1
    return call_price

#a)Download historical data for the S&P using the SPY ETF and for the VIX index, 
#which we will use as a proxy for volatility.
tickers1 = ['SPY']
tickers2 = ['^VIX']
SPY_ind = ETF(tickers1)
VIX_ind = ETF(tickers2)
SPY_data = SPY_ind.get_data("2010-01-01", "2020-06-30")
VIX_data = VIX_ind.get_data("2010-01-01", "2020-06-30")

#b)Examine both the S&P and the VIX index data for autocorrelation. 
def AR_test(data, nlags, title):
    ar_coef, _, p = stattools.acf(data, fft=False, nlags=nlags, qstat=True)
    ar_df = pd.DataFrame(ar_coef, columns=['AR_coeff'])
    p_df = pd.DataFrame(p, columns=['p_value'])
    return pd.concat([ar_df, p_df], axis=1)

def AR_plot(data, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_acf(data,ax=ax,color='blue')
    plt.title('Autocorrelation of ' + title, size = 25)
    plt.xlabel('time lags', size=20)
    plt.ylabel('AR coefficients', size=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    
SPY_AR = AR_test(SPY_data, 15, 'SPY')
VIX_AR = AR_test(VIX_data, 15, 'VIX')
print('\nAR_test for SPY index: \n',SPY_AR.head())
print('\nAR_test for VIX index: \n',VIX_AR.head())
AR_plot(SPY_data,'SPY')
AR_plot(VIX_data,'VIX')

#c)Calculate the correlation of the S&P and its implied volatility (using VIX as a proxy)
#on a daily and monthly basis.
SPY_corr_daily = SPY_data.corr(VIX_data)
print('\nthe correlation of the S&P and its implied volatility on daily basis:', SPY_corr_daily)
SPY_data_month = SPY_data.resample('M').ffill()
VIX_data_month = VIX_data.resample('M').ffill()
SPY_corr_monthly = SPY_data_month.corr(VIX_data_month)
print('the correlation of the S&P and its implied volatility on monthly basis:', SPY_corr_monthly)

#d)Calculate rolling 90-day correlatons of the S&P and its implied volatility as well.
SPY_data_corr_roll = SPY_data.rolling(90).corr(VIX_data).dropna()

plt.figure(figsize=(14, 8))
SPY_data_corr_roll.plot(color='red',label = 'Correlation')
plt.axhline(np.mean(SPY_data_corr_roll), color="blue",label = "Average Correlation",linewidth=2, linestyle='-.')
plt.title('Rolling coeffeicient of SPY and implied volatility', size=22)
plt.xlabel('Date', size=18)
plt.ylabel('Correlation coefficients', size=18)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()

#e)Calculate rolling 90-day realized volatilities in the S&P and compare them to the
#implied volatility (again using VIX as a proxy). Plot the premium of implied vol. over
#realized vol.
def realized_col(data, window = 90):
    temp = data**2
    re_vol = np.sqrt(temp.rolling(window).sum() * 252 / window) * 100
    return re_vol.dropna()

SPY_re_vol = realized_col(SPY_ind.return_data1(), window = 90)
vol_data = pd.merge(SPY_re_vol,VIX_data,left_index=True, right_index=True)
vol_data.columns = ['Realized','Implied']
premium_implied_vol = vol_data.Implied - vol_data.Realized

plt.figure(figsize=(14,8))
premium_implied_vol.plot(color = 'green')
plt.title('Premium of volatility', size=30)
plt.xlabel('Date', size=20)
plt.ylabel('Premium', size=20)
plt.xticks(size=18)
plt.yticks(size=18)

#f) Construct a portfolio that buys a 1M at-the-money straddle (long an at-the-money
#call and long an at-the-money put) every day in your historical period. Use the Black-
#Scholes model to compute the option prices and use the level of VIX as the implied vol
#input into the BS formula.

SPY_data = pd.DataFrame(SPY_data)
VIX_data = pd.DataFrame(VIX_data)
BSM_put(SPY_data.iloc[:, 0], SPY_data.iloc[:, 0], VIX_data.iloc[:, 0] / 100, 21 / 252, 0)
straddle = pd.merge(SPY_data, VIX_data / 100, left_index=True, on='Date')
straddle.columns = ['SPY','VIX']
straddle['Call Price'] = straddle.apply(lambda z: BSM_call(z[0], z[0], z[1], 21 / 252, 0), axis=1)
straddle['Put Price'] = straddle.apply(lambda z: BSM_put(z[0], z[0], z[1], 21 / 252, 0), axis=1)
print('\nprices of straddle: \n',straddle.head())

straddle.loc[:, ['Call Price']].plot(legend = 0,figsize = (14,8),color = 'blue')
plt.title("Option prices",size=30)
plt.xlabel('Date',size=20)
plt.ylabel('Prices',size=20)
plt.show()

#g)Calculate the payoffs of these 1M straddles at expiry (assuming they were held to
#expiry without any rebalances) by looking at the historical 1M changes in the S&P.
#Calculate and plot the P&L as well. What is the average P&L?
straddle[['at expiry 1M later']] = straddle[['SPY']].shift(-21)
straddle['payoff'] = np.abs(straddle.iloc[:, 4] - straddle.iloc[:, 0])
straddle['profit'] = straddle.iloc[:, 5] - straddle.iloc[:, 2] - straddle.iloc[:, 3]

straddle.loc[:, ['profit']].plot(legend = 0,figsize = (14,8))
plt.title("P&L of straddles",size=30)
plt.xlabel('Date',size=20)
plt.ylabel('P&L',size=20)
mean_PL = straddle.loc[:, ['profit']].mean()
plt.axhline(int(mean_PL), color="red",linewidth=2, linestyle='-')
plt.show()

print('\nstraddle info with profit and loss and payoff: \n', straddle.head())
#h)Make a scatter plot of this P&L against the premium between implied and realized
#volatility. Is there a strong relationship? Would you expect there to be a strong
#relationship? Why or why not?

premium_implied_vol = pd.DataFrame(premium_implied_vol)
premium_implied_vol.columns = ['Premium of volitility']
inte = pd.merge(straddle,premium_implied_vol,left_index=True,on='Date')

xx = inte.loc[:,['profit']]
yy = inte.loc[:,['Premium of volitility']]
reg_model = sm.OLS(yy,xx,missing = 'drop').fit()
reg_model_sum = reg_model.summary()
print(reg_model_sum)

fig = plt.scatter(inte.iloc[:, 7], inte.iloc[:, 6],marker = 'o', alpha=0.5, color = 'midnightblue')
plt.title('P&L against premium')
plt.xlabel('Premium of Volitility')
plt.ylabel('P&L')
plt.show()



