#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

#%%
# caras de un dado
die=pd.DataFrame([1,2,3,4,5,6])

# lanza 2 dados y sumalos
sum_of_dice=die.sample(2,replace=True).sum()[0]

print('La suma de los dados es: ',sum_of_dice)
# %%
trial=50
# repetir y anotar 50 veces
result=[die.sample(2,replace=True).sum()[0] for i in range(trial)]


# %%
# Frecuencia absoluta
freq=pd.DataFrame(result)[0].value_counts()

sort_freq=freq.sort_index() # ordenar de 2 a 12
relative_freq=sort_freq/trial # frecuencia relativa

# sort_freq.plot(kind='bar')
relative_freq.plot(kind='bar')
# %%
# para sacar la media desde la frecuencia
# haces el promedio ponderado segun la frecuencia de cada valor
# al final dividirias por 1, si es frecuencia relativa
mean=sum(relative_freq.index*relative_freq.values)
# np.mean(result)
varianze= sum(((relative_freq.index-mean)**2)*relative_freq.values)
# np.var(result)

print('Mean: ',mean,' Varianze: ', varianze)
# %%
from scipy.stats import norm
import numpy as np
# Data=pd.DataFrame()
# Data['LogReturn']=result
Data=pd.read_csv('Ejemplo1.csv')
Data['LogReturn'] = np.log(Data['close']).shift(-1) - np.log(Data['close'])

mu = Data['LogReturn'].mean()
sigma = Data['LogReturn'].std(ddof=1)

density=pd.DataFrame()
density['x']=np.arange(-0.0025,0.0025,0.00001)
density['pdf']=norm.pdf(density['x'],mu,sigma)
density['cdf']=norm.cdf(density['x'],mu,sigma)

Data['LogReturn'].hist(bins=30, figsize=(15, 8))
plt.plot(density['x'], density['pdf'], color='red')

# %%
# probability that the stock price of microsoft will drop over 5% in a day
prob_return1 = norm.cdf(-0.01, 24*mu, 24*sigma)
print('The Probability is ', prob_return1)
# %%
# prueba
plt.plot(density['x'], density['pdf'], color='red')
# plt.fill_between(x=np.arange(-0.1,-0.01,0.0001),y2=0,y1=norm.pdf(np.arange(-0.1,0.05,0.0001),mu,sigma),facecolor='pink',alpha=0.5)

print('5% quantile ', norm.ppf(0.05, mu, sigma))
# 95% quantile
print('95% quantile ', norm.ppf(0.95, mu, sigma))

# %%

# %%
