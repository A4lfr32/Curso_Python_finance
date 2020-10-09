#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Esto no lo estàs usando
import mplfinance as mpf
from matplotlib.dates import num2date

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

#veo el cierre de mañana
rates_frame['Price1']=rates_frame['close'].shift(-1)  

#El precio de cierre aumenta tal.. en la proxima hora
rates_frame['PriceDiff']=rates_frame['Price1']-rates_frame['close'] 

# si comprara cada hora
rates_frame['Return']=rates_frame['PriceDiff']/rates_frame['close']

#Rules, if  PriceDiff > 0 up, else down
rates_frame['Direction']=[1 if rates_frame.loc[ei,'PriceDiff']>0 else -1 for ei in rates_frame.index]

#%%
#Moving average
rates_frame['Average3']=(rates_frame.close+rates_frame.close.shift(-1)+rates_frame.close.shift(-2))/3

rates_frame['MA40']=rates_frame.close.rolling(40).mean()
rates_frame['MA200']=rates_frame.close.rolling(200).mean()

# %%
rates_frame.close.loc['2020-05':].plot()
rates_frame.MA40.loc['2020-05':].plot()
rates_frame.MA200.loc['2020-05':].plot()
# rates_frame[rates_frame['Share']==1].close.plot(marker='x')

# %%
# Estrategia 
rates_frame['Share']=[1 if rates_frame.loc[ei,'MA200']<rates_frame.loc[ei,'MA40'] else 0 for ei in rates_frame.index]

rates_frame['Close1']=rates_frame['close'].shift(-15)
rates_frame['Profit']=[rates_frame.loc[ei,'Close1']-rates_frame.loc[ei,'close'] if rates_frame.loc[ei,'Share']==1 else 0 for ei in rates_frame.index]

rates_frame['wealth']=rates_frame['Profit'].cumsum()

rates_frame['wealth'].plot()
print('Total money you win is ',rates_frame['wealth'][-2])
print('Total money you spent is ',rates_frame['close'][0])
# %%
rates_frame.to_csv('Ejemplo1.csv')

# %%
from scipy.fft import fft, ifft

signal_S=np.array(rates_frame.close)
signalmean=np.mean(signal_S)
signal=signal_S-signalmean
N=len(signal)
yf = fft(signal)
yf_nykist=yf
yf_nykist[:10]=0 #no recortes el tamaño, solo pon zeros

plt.plot(signal)
sol=ifft(yf_nykist)
plt.plot(np.linspace(0,N,len(sol)),sol)

# %%https://pythontic.com/visualization/signals/fouriertransform-ifft

# Do a Fourier transform on the signal
tx  = np.fft.fft(signal)
tx[-10:]  = 0
 
# Do an inverse Fourier transform on the signal
itx = np.fft.ifft(tx[:-1])

# Plot the original sine wave using inverse Fourier transform
plt.plot(signal)
plt.plot(np.linspace(0,len(signal),len(itx)),itx)

plt.title("Sine wave plotted using inverse Fourier transform")
plt.xlabel('Time')
plt.ylabel('Amplitude')
# plt.grid(True)
plt.show()

# %%
