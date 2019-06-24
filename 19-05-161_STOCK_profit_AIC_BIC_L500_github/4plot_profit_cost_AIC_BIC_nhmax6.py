import numpy as np
import matplotlib.pyplot as plt

#=========================================================================================
# average:
p1 = np.loadtxt('profit_cost_nhmax6.dat')
p2 = np.loadtxt('profit_AIC_nhmax6.dat')
p3 = np.loadtxt('profit_BIC_nhmax6.dat')

p4 = np.loadtxt('profit_nh4.dat')
p6 = np.loadtxt('profit_nh6.dat')
p8 = np.loadtxt('profit_nh8.dat')

tmax = np.shape(p1)[0]
t = np.arange(0,tmax,1)

plt.figure(figsize=(20,16))

plt.subplot(2,2,1)
#plt.figure(figsize=(5,4))

plt.title('trade everyday')
plt.plot(t, p1[:,0],'k-',label='cost')
plt.plot(t, p2[:,0],'b-',label='AIC')
plt.plot(t, p3[:,0],'r-',label='BIC')
plt.plot(t, p4[:,0],'r--',label='nh=4')
plt.plot(t, p6[:,0],'g--',label='nh=6')
plt.plot(t, p8[:,0],'o--',label='nh=8')
plt.legend()    
plt.xlabel('time')
plt.ylabel('cumulative profit')
plt.ylim([-1,4])
plt.grid(linestyle='dotted')


plt.subplot(2,2,2)
plt.title('not trade everyday')
plt.plot(t, p1[:,1],'k-',label='cost')
plt.plot(t, p2[:,1],'b-',label='AIC')
plt.plot(t, p3[:,1],'r-',label='BIC')
plt.plot(t, p4[:,1],'r--',label='nh=4')
plt.plot(t, p6[:,1],'g--',label='nh=6')
plt.plot(t, p8[:,1],'o--',label='nh=8')
plt.legend()    
plt.xlabel('time')
plt.ylabel('cumulative profit')
plt.ylim([-1,4])
plt.grid(linestyle='dotted')

#plt.tight_layout(h_pad=0.8, w_pad=1.2)
plt.savefig('profit_cost_AIC_BIC_nh468.pdf', format='pdf', dpi=300)
