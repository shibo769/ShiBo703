import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.graphics.gofplots import qqplot

class Bachelier:
    
    def __init__(self, r = 0, S0 = 100, sigma = 10.0, K = 100, T = 1, steps = 100, N = 10000):
        self.r = r
        self.S0 = S0
        self.sigma = 10.0
        self.K = 100
        self.T = 1
        self.steps = steps
        self.N = N
        
    def Simulation(self,S0):
        dt = self.T / self.steps
        self.simulate_all = []
        for i in range(self.N):
            simu = [S0]
            for j in range(self.steps):
                dW = np.random.normal(0,math.sqrt(dt))
                simu.append(self.sigma * dW + simu[j])
            self.simulate_all.append(simu)
        return pd.DataFrame(self.simulate_all).T
    
    def lookback_put_opt(self,S0):
        S_minimum = np.min(self.Simulation(S0))
        payoff = [np.maximum(self.K - S, 0) for S in S_minimum]
        Put_price = 1 / self.N * np.sum(payoff) * np.exp(-self.r * self.T)
        return Put_price
    
    def delta(self, epsilon):
        numerator = self.lookback_put_opt(self.S0 + epsilon) - self.lookback_put_opt(self.S0 - epsilon)
        denominator = 2 * epsilon
        return numerator / denominator
        
if __name__ == '__main__':
    #a)
    BBB = Bachelier()
    simulation = BBB.Simulation(100)
    plt.plot(simulation)
    plt.grid(True)
    plt.title('Simulation price')
    plt.xlabel('steps')
    plt.ylabel('Price by simulation')
    #b)
    ending_simulation = BBB.Simulation(100).iloc[-1,:]
    plt.figure()
    sns.distplot(ending_simulation,bins = 100,color = 'darkblue')
    plt.xlabel("ending value of simulation")
    plt.title("Histogram of ending value for simulations")
    #b)QQplot
    fig = plt.figure(figsize=(20,5))
    qqplot(ending_simulation,line='45',fit=True)
    plt.grid(True)
    plt.title('Normal test for ending prices')
    #c)
    put_price = BBB.lookback_put_opt(100)
    print('The price of lookback put option is',put_price)
    #d)
    eps_list = [0.01,0.02,0.03,0.04,0.1,0.2,0.3,0.5,0.6,0.7,1,2,3,4,5,6,7,8,9,10]
    list_delta = []
    for i in eps_list:
        list_delta.append(BBB.delta(i))
    plt.figure()
    plt.grid(True)
    plt.plot(eps_list, list_delta,linewidth=2.5)
    plt.axhline(y = -1,color='red',ls='-.', linewidth = 3)
    plt.title('Delta v.s eps',size=25)
    plt.ylabel('delta',size=20)
    plt.xlabel('eps',size=20)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    