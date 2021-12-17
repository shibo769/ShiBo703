import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf

# 1a
# df of etf data
yf.pdr_override()
etf_name = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
etf_df = []
for etf in etf_name:
    etf_df.append(pdr.get_data_yahoo(etf, start="2010-01-01", end="2020-12-04"))


# 1b
# return of etf
etf_ret = []
for df in etf_df:
    cls = df["Adj Close"].values.tolist()
    ret = []
    for i in range(1,len(cls)):
        ret.append((cls[i] - cls[i-1])/cls[i-1])
    etf_ret.append(ret)
etf_ret = np.array(etf_ret)

# cor matrix of return
etf_cov = np.cov(etf_ret)
sns.set(font_scale=0.7)
plt.figure(dpi = 100)
sns.heatmap(etf_cov, xticklabels = etf_name,yticklabels = etf_name,annot=True, cmap = 'YlGnBu')
plt.title('Q1(b):covariance matrix of ETF daily return')


# 1c
# eigenvalue decomposition
eigvalue = np.sort(np.linalg.eig(etf_cov)[0])[::-1]
plt.figure()
plt.plot(np.arange(1,10),eigvalue)
plt.ylabel("eigenvalue")
plt.title("Q1(c):eigenvalues of ETF covariance matrix")


# 1d
# random matrix
rand_mat = np.random.randn(9, 9)
sns.set(font_scale=0.8)
plt.figure(dpi = 100)
sns.heatmap(rand_mat, xticklabels = etf_name,yticklabels = etf_name,annot=True)
plt.title('Q1(d):random normal matrix')

# 1e
# eigenvalue decomposition
eigvalue_rand = np.sort(np.linalg.eig(rand_mat)[0])[::-1]
plt.figure()
plt.plot(np.arange(1,10),eigvalue_rand)
plt.ylabel("eigenvalue")
plt.title("Q1(e):eigenvalues of random normal matrix")




# 2a
# ann return
ann_return = [] 
year_length = etf_ret.shape[1]/250
for i in range(etf_ret.shape[0]):
    multi = 1 + etf_ret[i,0]
    for j in range(1,etf_ret.shape[1]): 
        multi = multi * (1 + etf_ret[i,j])
    ann_return.append(multi**(1/year_length)-1)
ann_return = np.array(ann_return)
plt.figure()
plt.xticks(np.arange(9), etf_name, rotation = 45, fontsize = 8)
plt.bar(np.arange(9), height= ann_return)
plt.ylabel("annualized return")
plt.title("Q2(a):annualized return of ETF")
    

# 2b
# weights of portfolio
weight = 0.5 * np.linalg.inv(etf_cov) @ ann_return
weight = weight/np.sum(weight)
plt.figure()
plt.xticks(np.arange(9), etf_name, rotation = 45, fontsize = 10)
plt.bar(np.arange(9), height= weight)
plt.ylabel("weights")
plt.title("Q2(b):weights of portfolio")
    
    
# 2c
# weights by adj return
sigma_list = [0.005,0.01,0.05,0.1]
def adj_weight(sigma_list,cov):
    weight = []
    for sigma in sigma_list:
        adj_return = np.array([u + sigma * np.random.normal() for u in ann_return])
        w = 0.5 * np.linalg.inv(cov) @ adj_return
        w = w / np.sum(w)
        weight.append(w)
    return np.array(weight)
weight1 = adj_weight(sigma_list,etf_cov)
    
fig = plt.figure()
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    plt.xticks(np.arange(9), etf_name, rotation = 45, fontsize = 10)
    plt.bar(np.arange(9), height= weight1[i-1])
    plt.ylabel("weights")
    ax.text(0,0.1,"sigma:"+str(sigma_list[i-1]))


# 2d
# diag
diag = []
for i in range(9):
    diag.append(np.var(etf_ret[i,:]))
diag_mat = np.diag(diag)

# weights by regularized covarinace matrix
def reg_mat(delta):
    mat = delta * diag_mat + (1 - delta) * etf_cov
    return mat
    


# 2e
eigvalue1 = np.sort(np.linalg.eig(reg_mat(1))[0])[::-1]
plt.figure()
plt.plot(np.arange(1,10),eigvalue1)
plt.ylabel("eigenvalue")
plt.title("Q2(e):eigenvalues of regularized covariance matrix, delta = 1")


# 2f
delta_list = [0.25,0.5,0.75,1]
fig = plt.figure()
for i in range(1,5):
    eigvalue2 = np.sort(np.linalg.eig(reg_mat(delta_list[i-1]))[0])[::-1]
    ax = fig.add_subplot(2, 2, i)
    plt.plot(np.arange(1,10),eigvalue2)
    plt.ylabel("eigenvalue")
    plt.title("delta"+str(delta_list[i-1]))
    

# 2g
for delta in delta_list:
    fig = plt.figure()
    plt.title("delta:"+str(delta))
    weight2 = adj_weight(sigma_list,reg_mat(delta))
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        plt.xticks(np.arange(9), etf_name, rotation = 45, fontsize = 7)
        plt.bar(np.arange(9), height= weight2[i-1])
        plt.ylabel("weights")
        ax.text(0,0.1,"sigma:"+str(sigma_list[i-1]))
        