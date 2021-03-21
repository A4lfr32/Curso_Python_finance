#%%
import numpy as np
import pandas as pd
from mytools import historyData
import MetaTrader5 as mt5

# %%
def calcCorr(str1,str2):
    d1=historyData(str1)
    d2=historyData(str2)
    # print(f'{str1} vs {str2} = {pd.Series.corr(d1.Close,d2.Close)}')
    return pd.Series.corr(d1.Close,d2.Close)

# %%
FXmajor=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCHF','NZDUSD','USDCAD']
# %%
table=pd.DataFrame(columns= FXmajor, index=FXmajor)
for i,x in enumerate(FXmajor):
    for j,y in enumerate(FXmajor):
        table.iloc[i,j]=calcCorr(x,y)
table
# %%

# %%
