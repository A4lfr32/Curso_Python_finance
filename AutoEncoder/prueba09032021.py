#%% Parte 1: Obtención de datos

import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mytools import historyData

#%%

data=historyData('EURUSD')
data
# %% Parte 2: preprocesamiento de entrada y salida
# thisData=data.Close.rolling(10).std().values
# thisData=thisData[~np.isnan(thisData)]
# %%
nTicks=10
ticksLookAhead=3
dataset = tf.data.Dataset.from_tensor_slices(data.Close.values)
# dataset = tf.data.Dataset.range(100)
dataset = dataset.window(nTicks+ticksLookAhead, shift=ticksLookAhead, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(13))
dataset = dataset.map(lambda window: (tf.expand_dims(window[:-ticksLookAhead], axis=0), window[-ticksLookAhead:]))
dataset = dataset.shuffle(buffer_size=13)
full_dataset=dataset.batch(10).prefetch(1)

dSize=len(list(full_dataset))
train_dataset = full_dataset.take(dSize-10)
test_dataset = full_dataset.skip(dSize-10)
# val_dataset = test_dataset.skip(7)
# test_dataset = test_dataset.take(7)
for x,y in train_dataset:
  print(x.numpy().shape, y.numpy())
# %%
model = tf.keras.Sequential([
  # tf.keras.layers.Dense(6, activation=tf.nn.relu,input_shape=(None,10)),  # input shape required
  tf.keras.layers.SimpleRNN(10, activation=tf.nn.relu,input_shape=(None,10)),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(3)
])
model.summary()
# %%
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9))
history = model.fit(train_dataset,validation_data=test_dataset,epochs=50,verbose=2)

# %%
datasetP = tf.data.Dataset.from_tensor_slices(thisData[-100:])
# dataset = tf.data.Dataset.range(100)
datasetP = datasetP.window(nTicks, shift=None, drop_remainder=True)
datasetP = datasetP.flat_map(lambda window: window.batch(100))
datasetP = datasetP.map(lambda window: tf.expand_dims(window[-100:], axis=0))
datasetP = datasetP.shuffle(buffer_size=100)
datasetP=datasetP.batch(5).prefetch(1)

y=model.predict(datasetP)

# %%
plt.plot(range(100),thisData[-100:])
plt.plot(range(100,130),y.flatten())
# %%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
