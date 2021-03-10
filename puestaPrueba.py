#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Esto no lo estàs usando
import mplfinance as mpf
from matplotlib.dates import num2date
from scipy.stats import norm

# Initializing MT5 connection 
import MetaTrader5 as mt5
mt5.initialize()
# mt5.WaitForTerminal()


print(mt5.terminal_info())
print(mt5.version())

# Copying data to pandas data frame
stockdata = pd.DataFrame()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 5000)
# Deinitializing MT5 connection
mt5.shutdown()
#%%
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the datetime format
rates_frame.index=pd.to_datetime(rates_frame['time'], unit='s')


#%%

Only_two_columns=rates_frame[['close','open']]

#Moving average
rates_frame['min15']=-Only_two_columns.close+Only_two_columns.close.shift(-1)

# %%

freq=rates_frame['min15'].value_counts()
sort_freq=freq.sort_index()
relative_freq=sort_freq/len(freq) # frecuencia relativa

rates_frame['min15'].hist(bins=40)
mu=rates_frame['min15'].mean()
sigma=rates_frame['min15'].std(ddof=1)

plt.plot(np.arange(-0.005,0.005,0.0001),norm.pdf(np.arange(-0.005,0.005,0.0001),mu,sigma),color='red')
print(mu,sigma)
# %%
# probability that the stock price of microsoft will drop over 5% in a day
prob_return1 = norm.cdf(-0.0, 1*mu, 1*sigma)
print('The Probability is ', prob_return1)
# %%
