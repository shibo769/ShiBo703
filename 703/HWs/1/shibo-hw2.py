# -*- coding: utf-8 -*-
"""
Assignment 1 for MF 703: Programming for Mathematical Finance
Professor: Chris Kelliher
Author: SHI BO
Email: shibo@bu.edu
Date: 09-07-2020
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulation(s0, r, sigma, T, n, is_show=False):
    ''' dst = r * St * dt + sigma * St * dWt, 1 day every step, input:
        s0: intial asset price
        r: trend rate(annual)
        sigma: volality(annual)
        T: time range(year)
        n: number of paths
        is_show: draw picture of path or not, defalut is not.
    ''' 
    st = []
    ssmin = []
    spath = {}
    for i in range(n):
        daily_returns = np.random.normal(r/252, sigma/np.sqrt(252), 252*T)
        price_list = [s0]
        for x in daily_returns:
            price_list.append(price_list[-1]*(x+1))
        plt.plot(price_list)
        st.append(price_list[-1])
        ssmin.append(min(price_list))
        spath[i] = price_list
    if is_show == True: 
        plt.xlabel('time')
        plt.ylabel('price')
        plt.title('Random walk of price')
        plt.show()
    return spath, st, ssmin


if __name__ == '__main__':
# question 1
    s0 = 100
    r = 0
    sigma = 0.25
    T = 1
    n = 1000
    spath, st, ssmin = simulation(s0, r, sigma, T, n, True)
    st_mean = np.mean(st)
    st_var = np.var(st)
    print("St mean is ", st_mean, ",\nSt var is ", st_var)
    # theoretically, the mean should be 100, and the var should be 625.
    # The result is almost consist with the theory.
# question 2
    k = 100
    payoff_put_opt = np.maximum(k - np.array(st), 0)
    plt.hist(payoff_put_opt)
    plt.xlabel('payoff')
    plt.ylabel('frequency')
    plt.title('payoff of European Option')
    plt.show()
    payoff_mean = np.mean(payoff_put_opt)
    payoff_std = np.std(payoff_put_opt)
    print("payoff mean is", payoff_mean, ",\npayoff std is ", payoff_std)
# question 3
    dis_price_put_opt = np.mean(payoff_put_opt) / (1+r)
    print("price of put option based on averagely discount is ", dis_price_put_opt)
# question 4
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    bsm_price_put_opt = k * np.exp(-r * T) * stats.norm.cdf(-d2) - s0 * stats.norm.cdf(-d1)
    print("price of put option based on BS formula is ", bsm_price_put_opt)
    diff = dis_price_put_opt - bsm_price_put_opt
    print("the difference between these two methods is ", diff)
    # the prices based on average discount and on BS formula are similar
# question 5
    payoff_lookback = np.maximum(0, k - np.array(ssmin))
    lookback_price = np.mean(payoff_lookback)/(1+r)    
    print("price of lookback put option is ", lookback_price)
# question 6
    premium = payoff_lookback - payoff_put_opt
    print('the highest premium is', max(premium),'and the lowest premium is', min(premium))
    print('It is obvious that the premium can not be negative.')
    plt.plot(spath[np.argmax(premium)],label = 'highest premium')
    plt.plot(spath[np.argmin(premium)],label = 'lowest premium')
    plt.legend()
    plt.show()
    # the highest premium occurs when asset price goes down hugely,
    # but eventually turns back quite a lot at expiracy.
    # the lowest premium occurs when st is actually the lowest price.
    # the premium can never be negative because min(spath) >= st
# question 7
    '''sigma = 0.5'''
    sigma1 = 0.5
    spath1, st1, ssmin1 = simulation(s0, r, sigma1, T, n)
    payoff_put_opt1 = np.maximum(k - np.array(st1), 0)
    dis_price_put_opt1 = np.mean(payoff_put_opt1) / (1+r)
    print("As the new sigma = 0.5, price of put option based on averagely discount is", dis_price_put_opt1)
    d11 = (np.log(s0 / k) + (r + 0.5 * sigma1 ** 2) * T) / (sigma1 * np.sqrt(T))
    d21 = (np.log(s0 / k) + (r - 0.5 * sigma1 ** 2) * T) / (sigma1 * np.sqrt(T))
    bsm_price_put_opt1 = k * np.exp(-r * T) * stats.norm.cdf(-d21) - s0 * stats.norm.cdf(-d11)
    print("As the new sigma = 0.5, price of put option based on BS formula is ", bsm_price_put_opt1)
    payoff_lookback1 = np.maximum(0, k - np.array(ssmin1))
    lookback_price1 = np.mean(payoff_lookback1)/(1+r) 
    print("As the new sigma = 0.5, price of lookback put option is", lookback_price1)
    premium1 = payoff_lookback1 - payoff_put_opt1
    print('As the new sigma = 0.5, the highest premium is', max(premium1),'and the lowest premium is', min(premium1))
    '''sigma = 0.75'''
    sigma2 = 0.75
    spath2, st2, ssmin2 = simulation(s0, r, sigma2, T, n)
    payoff_put_opt2 = np.maximum(k - np.array(st2), 0)
    dis_price_put_opt2 = np.mean(payoff_put_opt2) / (1+r)
    print("As the new sigma = 0.75, price of put option based on averagely discount is", dis_price_put_opt2)
    d12 = (np.log(s0 / k) + (r + 0.5 * sigma2 ** 2) * T) / (sigma2 * np.sqrt(T))
    d22 = (np.log(s0 / k) + (r - 0.5 * sigma2 ** 2) * T) / (sigma2 * np.sqrt(T))
    bsm_price_put_opt2 = k * np.exp(-r * T) * stats.norm.cdf(-d22) - s0 * stats.norm.cdf(-d12)
    print("As the new sigma = 0.75, price of put option based on BS formula is ", bsm_price_put_opt2)
    payoff_lookback2 = np.maximum(0, k - np.array(ssmin2))
    lookback_price2 = np.mean(payoff_lookback2)/(1+r) 
    print("As the new sigma = 0.75, price of lookback put option is", lookback_price2)
    premium2 = payoff_lookback2 - payoff_put_opt2
    print('As the new sigma = 0.75, the highest premium is', max(premium2),'and the lowest premium is', min(premium2))
    '''sigma = 1.00'''
    sigma3 = 1
    spath3, st3, ssmin3 = simulation(s0, r, sigma2, T, n)
    payoff_put_opt3 = np.maximum(k - np.array(st3), 0)
    dis_price_put_opt3 = np.mean(payoff_put_opt3) / (1+r)
    print("As the new sigma = 1.00, price of put option based on averagely discount is", dis_price_put_opt3)
    d13 = (np.log(s0 / k) + (r + 0.5 * sigma3 ** 2) * T) / (sigma3 * np.sqrt(T))
    d23 = (np.log(s0 / k) + (r - 0.5 * sigma3 ** 2) * T) / (sigma3 * np.sqrt(T))
    bsm_price_put_opt3 = k * np.exp(-r * T) * stats.norm.cdf(-d23) - s0 * stats.norm.cdf(-d13)
    print("As the new sigma = 1.00, price of put option based on BS formula is ", bsm_price_put_opt3)
    payoff_lookback3 = np.maximum(0, k - np.array(ssmin3))
    lookback_price3 = np.mean(payoff_lookback3)/(1+r) 
    print("As the new sigma = 1.00, price of lookback put option is", lookback_price3)
    premium3 = payoff_lookback3 - payoff_put_opt3
    print('As the new sigma = 1.00, the highest premium is', max(premium3),'and the lowest premium is', min(premium3))
    # if increasing the sigma, the price of options would increase, the premium would increase,
    # is decreasing the sigma, the price of options would decrease, the premium would decrease.
    