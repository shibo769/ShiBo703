from ETF import *
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt  
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from pyfinance.ols import PandasRollingOLS
from scipy.stats import kstest

#Question A
df_FF = pd.read_csv('F-F_Research_Data_Factors_daily.csv',
                    index_col=0,
                    parse_dates=True,
                    dayfirst=True)
del df_FF['RF']
df_FF.index.name = 'date'
df_FF = df_FF.loc[df_FF.index.year >= 2010]
df_FF = df_FF.iloc[1:2640,:]
validate_data = df_FF.isna().sum()
print('Chech whether there are anomalies:\n', validate_data)

#Question B
df_FF_corr = df_FF.corr()
aa = sns.heatmap(df_FF_corr, annot = True)
plt.figure(figsize = (14, 10), dpi = 100)
sns.set(font_scale = 1)
aa.set_title('correlation matrix of the factor returns',fontsize = 20)
plt.show()
df_FF_cov = df_FF.cov()
bb = sns.heatmap(df_FF_cov, annot = True)
plt.figure(figsize = (14,10), dpi = 100)
sns.set(font_scale = 1)
bb.set_title('covariance matrix of the factor returns', fontsize = 20)
plt.show()

#Question C
df_FF_corr_roll = df_FF.rolling(90).corr().dropna()

location1 = df_FF_corr_roll.index.get_level_values(1) == 'Mkt-RF'
mktrf_smb = df_FF_corr_roll[location1][['SMB']]
mktrf_hml = df_FF_corr_roll[location1][['HML']]
location2 = df_FF_corr_roll.index.get_level_values(1) == 'SMB'
smb_hml = df_FF_corr_roll[location2][['HML']]

df_FF_corr_roll = pd.DataFrame(
    data=np.hstack((mktrf_smb.values,mktrf_hml.values,smb_hml.values)),
    index=mktrf_smb.index.droplevel(level=1),
    columns=['Mkt-RF and SMB', 'Mkt-RF and HML', 'SMB and HML'])

df_FF_corr_roll.plot(figsize = (20,5))
plt.xlabel('Date',fontsize = 15)
plt.ylabel('correlation', fontsize = 15)
plt.title('90-day corrlation of each factor', fontsize = 15)
plt.show()

#Question D
fig = plt.figure(figsize=(20,5))
fig.suptitle('Qqplot Normal Test',fontsize=15)
ax1 = plt.subplot(131)
ax1.set_title('Mkt-RF')
qqplot(df_FF['Mkt-RF'].values, ax=ax1,line='45',fit=True)
ax2 = plt.subplot(132)
ax2.set_title('SMB')
qqplot(df_FF['SMB'].values, ax=ax2,line='45',fit=True)
ax3 = plt.subplot(133)
ax3.set_title('HML')
qqplot(df_FF['HML'].values, ax=ax3,line='45',fit=True)
plt.show()

#Question E
tickers = ['SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
ETFs = ETF(tickers)
price_data = ETFs.get_data("2010-01-01", "2020-06-30")
ETF_return = ETFs.return_data1()
ETF_return = ETF_return*100

def Linear_reg(X, y, return_value):
    df = pd.merge(X, y, left_index=True, right_index=True)
    est = sm.OLS(df.iloc[:,-1], df.iloc[:,:3], missing='drop')
    result = est.fit()
    if return_value == 'beta':
        return result.params
    elif return_value == 'residual':
        return result.resid #y - est.predict()
beta = ETF_return.apply(lambda x: Linear_reg(df_FF , x , return_value='beta'))
print(beta.head)

beta_result = {}
plt.figure(figsize = (30,15),dpi=120)
for i, c in enumerate(ETF_return):
    plt.subplot(5,2,i+1)
    plt.grid(True)
    plt.title(tickers[i], fontsize = 25)
    plt.tight_layout()
    model = PandasRollingOLS(y=ETF_return[c], x=df_FF, window=90)
    beta_result[c] = model.beta
    model.beta.plot(ax = plt.gca())

#Question F
residual_dict = {}
residual_test = {}
for c in ETF_return:
    residual_dict[c] = Linear_reg(ETF_return[c], df_FF, return_value = 'residual')
    test_stat = kstest(residual_dict[c], 'norm')
    residual_test[c] = test_stat.pvalue
    
for i in residual_dict:
    print('The residual mean for',i,'is: ',residual_dict[i].mean())
    
for i in residual_dict:
    print('The residual variance for',i,'is: ',residual_dict[i].var())

for i, j in residual_test.items():
    print('index: {}, p_value: {}'.format(i, j))
 


