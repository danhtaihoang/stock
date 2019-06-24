import numpy as np
import matplotlib.pyplot as plt

#=========================================================================================
# average:
p = np.loadtxt('profit_av_non-predict.dat')
p_nh0 = np.loadtxt('profit_nh0.dat')
p_nh = np.loadtxt('profit_nh.dat')

tmax = np.shape(p)[0]
t = np.arange(0,tmax,1)

plt.figure(figsize=(20,16))

plt.subplot(2,2,1)
#plt.figure(figsize=(5,4))

plt.title('trade everyday')
plt.plot(t, p[:,0],'k-',label='w/o predict')
plt.plot(t, p_nh0[:,0],'b-',label='nh=0')
plt.plot(t, p_nh[:,0],'r-',label='nh')
plt.legend()    
plt.xlabel('time')
plt.ylabel('cumulative profit')
plt.ylim([-1,4])
plt.grid(linestyle='dotted')


plt.subplot(2,2,2)
plt.title('not trade everyday')
plt.plot(t, p[:,1],'k-',label='w/o predict')
plt.plot(t, p_nh0[:,1],'b-',label='nh=0')
plt.plot(t, p_nh[:,1],'r-',label='nh')
plt.legend()    
plt.xlabel('time')
plt.ylabel('cumulative profit')
plt.ylim([-1,4])
plt.grid(linestyle='dotted')


#plt.tight_layout(h_pad=0.8, w_pad=1.2)
plt.savefig('profit.eps', format='eps', dpi=300)
