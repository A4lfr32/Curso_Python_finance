#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# %%
import MetaTrader5 as mt5
mt5.initialize()
# mt5.WaitForTerminal()


print(mt5.terminal_info())
print(mt5.version())

# Copying data to pandas data frame
stockdata = pd.DataFrame()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10000)
# Deinitializing MT5 connection
mt5.shutdown
# %%
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the datetime format
rates_frame.index=pd.to_datetime(rates_frame['time'], unit='s')

# %%
Only_two_columns=rates_frame[['close','open']]

#veo el cierre de mañana
rates_frame['closeTomorrow']=rates_frame['close'].shift(-1)  

#El precio de cierre aumenta tal.. en la proxima hora
rates_frame['winOneDay']=rates_frame['closeTomorrow']-rates_frame['close'] 

# si comprara cada hora
rates_frame['Return']=rates_frame['winOneDay']/rates_frame['close']

#Rules, if  winOneDay > 0 up, else down
rates_frame['UpDown']=[1 if rates_frame.loc[ei,'winOneDay']>0 else -1 for ei in rates_frame.index]

# %%
#Moving average
rates_frame['MA40']=rates_frame.close.rolling(40).mean()
rates_frame['MA200']=rates_frame.close.rolling(200).mean()
# %%
# rates_frame.close.loc['2020-05':].plot()
rates_frame.close.plot()
rates_frame.MA40.plot()
rates_frame.MA200.plot()
# %%
rates_frame['Share']=[1 if rates_frame.loc[ei,'MA200']<rates_frame.loc[ei,'MA40'] else 0 for ei in rates_frame.index]

rates_frame['Close1']=rates_frame['close'].shift(-1)
rates_frame['Profit']=[rates_frame.loc[ei,'Close1']-rates_frame.loc[ei,'close'] if rates_frame.loc[ei,'Share']==1 else 0 for ei in rates_frame.index]

rates_frame['wealth']=rates_frame['Profit'].cumsum()

rates_frame['wealth'].plot()
print('Total money you win is ',rates_frame['wealth'][-2])
print('Total money you spent is ',rates_frame['close'][0])

# %%
from scipy.stats import norm
import numpy as np
Data=rates_frame
Data['LogReturn'] = np.log(Data['close']).shift(-1) - np.log(Data['close'])

mu = Data['LogReturn'].mean()
sigma = Data['LogReturn'].std(ddof=1)

density=pd.DataFrame()
density['x']=np.arange(-4,4,0.00001)
density['pdf']=norm.pdf(density['x'],0,1)
density['cdf']=norm.cdf(density['x'],0,1)

density = pd.DataFrame()
density['x'] = np.arange(Data['LogReturn'].min()-0.0001, Data['LogReturn'].max()+0.01, 0.00001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

Data['LogReturn'].hist(bins=100, figsize=(15, 8))
plt.plot(density['x'], density['pdf'], color='red')
# %%
# probability that the stock price of microsoft will drop over 5% in a day
prob_return1 = norm.cdf(-0.05, 24*mu, 24*sigma)
print('The Probability is ', prob_return1*100,'%')
prob_return1 = norm.cdf(0.05, 24*mu, 24*sigma)
print('The Probability is ', 100-prob_return1*100,'%')
# %%
# prueba
plt.plot(density['x'], density['pdf'], color='red')
# plt.fill_between(x=np.arange(-0.1,-0.01,0.0001),y2=0,y1=norm.pdf(np.arange(-0.1,0.05,0.0001),mu,sigma),facecolor='pink',alpha=0.5)

print('5% quantile ', norm.ppf(0.05, mu, sigma))
# 95% quantile
print('95% quantile ', norm.ppf(0.95, mu, sigma))

# %%
