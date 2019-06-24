import numpy as np
#import sys
#from scipy import stats
import matplotlib.pyplot as plt

ds0 = np.loadtxt('close_open_percent.txt')

n = ds0.shape[1]

t2min = 754
t2max = 3394
ds = ds0[t2min:t2max]

profit = np.cumsum(ds,axis=0)
profit_av = np.mean(profit,axis=1)
profit_dev = np.std(profit,axis=1)

#print(profit)
print(profit_av)
print(profit_dev)

#=========================================================================================
# trade on only some days:
profit2 = np.zeros((t2max-t2min,n))
itrade = 0 ; it = 0

for t2 in range(t2min,t2max):
    for i in range(n):        
        if ds0[t2-1,i] < 0 :    
            profit2[it,i] = ds0[t2,i]
        else:
            profit2[it,i] = - ds0[t2,i]

    it +=1

p2_cum = np.cumsum(profit2,axis=0)
profit2_av = p2_cum.mean(axis=1)

np.savetxt('profit_non-predict.dat',profit,fmt=' %3.5f')
np.savetxt('profit_av_non-predict.dat',zip(profit_av,profit2_av),fmt=' %3.5f')

plt.plot(profit_av)
plt.plot(profit2_av)






