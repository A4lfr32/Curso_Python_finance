#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
signalIn=np.random.randn(1000000,2)

signal=np.sum(np.random.randn(1000000,2),axis=1)
plt.hist(signal,bins=30)

# %%
n=5000000
count=0

for _ in range(n):
    inp=np.random.randn()
    if inp>=1:
        count+=1
    else:
        inp+=np.random.randn()
        if inp>=1:
            count+=1
        else:
            inp+=np.random.randn()
            if inp>=1:
                count+=0

print(count/n)
# %%
n=5000000
count=0

for _ in range(n):
    inp=np.random.randn()+np.random.randn()+np.random.randn()
    if inp>1:
        count+=1
print(count/n)