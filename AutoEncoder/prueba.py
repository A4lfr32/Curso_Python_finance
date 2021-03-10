#%%
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
# import datetime
from datetime import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

#%%
def historyData(Symbol):
    mt5.initialize(server="ForexClub-MT5 Demo Server",login=500063649,password="hrOmcAAn")
    # print(mt5.terminal_info())
    # print(mt5.version())
    listSymbols=mt5.symbols_get()
    # [x.name for x in listSymbols]
    # Symbol=np.random.choice(FXmajor, 1)[0]
    print(Symbol)
    pointValue=mt5.symbol_info(Symbol).point
    # mt5.Buy("EURUSD", 0.1,price=11395,ticket=9)
    Num_velas=1000
    # Copying data to pandas data frame
    # rates =  mt5.copy_rates_from_pos(Symbol, mt5.TIMEFRAME_M1, 0, Num_velas)
    rates =  mt5.copy_rates_range(Symbol, mt5.TIMEFRAME_H1, datetime(2021, 1, 10), datetime.now())
    # rates =  mt5.copy_rates_range("ES", mt5.TIMEFRAME_D1, datetime(2019, 1, 15), datetime(2019, 1, 25))
    # rates =  mt5.copy_rates_from_pos(Symbol, mt5.TIMEFRAME_M1, 0, Num_velas)

    # Deinitializing MT5 connection
    mt5.shutdown()
    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame.index=pd.to_datetime(rates_frame['time'], unit='s')

    rates_frame.columns=['time', 'Open', 'High', 'Low', 'Close', 'tick_volume', 'spread','real_volume']
    return rates_frame

#%%
window_length=50
encoding_dim=3
# this is our input placeholder
input_window = Input(shape=(window_length,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_window)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(window_length, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_window, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_window, encoded)


autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#%%
data=historyData('EURUSD')

scaler = MinMaxScaler()
x_train_nonscaled = np.array([data['Close'].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length+1,len(data['Close'])))])
x_train = np.array([scaler.fit_transform(data['Close'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length+1,len(data['Close'])))])

test_samples=40
x_test = x_train[-test_samples:]
x_train = x_train[:-test_samples]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train_simple = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_simple = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#%%
epochs=1000
history = autoencoder.fit(x_train_simple, x_train_simple,
                epochs=epochs,
                batch_size=24,
                shuffle=True,
                validation_data=(x_test_simple, x_test_simple))

decoded_stocks = autoencoder.predict(x_test_simple)
# %%
