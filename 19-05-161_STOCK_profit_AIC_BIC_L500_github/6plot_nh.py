import numpy as np
import matplotlib.pyplot as plt

#=========================================================================================
# average:
nh = np.loadtxt('nh_501.dat')

bins = np.linspace(-0.5,8.5,9, endpoint=False)

plt.figure(figsize=(20,16))

plt.subplot(2,2,1)
#plt.figure(figsize=(5,4))
plt.title('nh')
plt.plot(nh[:,0],'ko')
#plt.legend()    
plt.xlabel('time')
plt.ylabel('nh')
plt.ylim([0,7])
plt.grid(linestyle='dotted')


plt.subplot(2,2,2)
plt.hist(nh[:,0], bins, histtype='bar',rwidth=0.8)
plt.xlabel('number of hidden variables')
plt.ylabel('histogram')

#plt.tight_layout(h_pad=0.8, w_pad=1.2)
plt.savefig('nh.eps', format='eps', dpi=300)
