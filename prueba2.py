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
Only_one_columns=pd.DataFrame()
new=pd.DataFrame()
Only_one_columns['actual']=rates_frame['close']
new['tomorrow']=rates_frame['close'].shift(-24)


Only_one_columns['MA5']=rates_frame.close.rolling(10).mean()
Only_one_columns['MA10']=rates_frame.close.rolling(50).mean()
Only_one_columns['MA15']=rates_frame.close.rolling(100).mean()
Only_one_columns['MA20']=rates_frame.close.rolling(200).mean()

# %%
rates_frame['Direction']=[new.loc[ei,'tomorrow']-Only_one_columns.loc[ei,'actual'] for ei in rates_frame.index]

Only_one_columns=Only_one_columns.fillna(0)
rates_frame['Direction']=rates_frame['Direction'].fillna(0)
# %%
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()  # create object for the class

linear_regressor.fit(Only_one_columns,rates_frame['Direction'])

# %%
linear_regressor.score(Only_one_columns, rates_frame['Direction'])
# %%
predicted=linear_regressor.predict(Only_one_columns)
# %%
plt.plot(np.arange(len(rates_frame['Direction'][:-1])),rates_frame['Direction'][:-1],'.')
plt.plot(np.arange(len(predicted[:-1])),predicted[:-1])
plt.show()
# %%
plt.plot(np.arange(len(rates_frame['Direction'][:-1])),rates_frame['Direction'][:-1],'.')
plt.plot(rates_frame.close.rolling(10).std().fillna(0).values)
# %%
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.python import keras

model = Sequential()
model.add(Dense(10, activation='relu',input_dim=5))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
# %%
X=Only_one_columns.to_numpy().reshape((-1,5))
y=rates_frame['Direction'].to_numpy().reshape((-1,1))

model.fit(X,y,epochs=100,batch_size=128)

# %%
result=model.predict(X)
# %%
plt.plot(y)

plt.plot(result)
# %%
# %%
