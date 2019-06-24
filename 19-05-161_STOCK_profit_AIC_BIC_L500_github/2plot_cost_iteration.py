import numpy as np
import matplotlib.pyplot as plt

nh=3


#a = np.loadtxt('C/C_%02d.dat'%nh)
#c0 = a[:,2]
#c = a[:,3]

#plt.gca().set_color_cycle(['black', 'blue', 'red', 'orange','magenta'])

iloop = np.arange(0,500,1)  

plt.figure(figsize=(20,16))

nh_list = [1,2,3,4,5,6,7,8]

cost_obs = np.empty((500,len(nh_list)+1))
cost = np.empty((500,len(nh_list)+1))
for i,nh in enumerate(nh_list):
    a = np.loadtxt('cost/cost_%02d.dat'%nh)
    cost_obs[:,nh] = a[:,0]
    cost[:,nh] = a[:,1]

plt.subplot(3,3,1)
plt.title('obs')
plt.plot(iloop, cost_obs[:,1], 'k', label='nh=1' )
plt.plot(iloop, cost_obs[:,2], 'b', label='nh=2')
plt.plot(iloop, cost_obs[:,3], 'r', label='nh=3')
plt.plot(iloop, cost_obs[:,4], 'k--', label='nh=4')
plt.plot(iloop, cost_obs[:,5], 'b--', label='nh=5')
plt.plot(iloop, cost_obs[:,6], 'r--', label='nh=6')

plt.subplot(3,3,2)
plt.title('entire')
plt.plot(iloop, cost[:,1], 'k', label='nh=1' )
plt.plot(iloop, cost[:,2], 'b', label='nh=2')
plt.plot(iloop, cost[:,3], 'r', label='nh=3')
plt.plot(iloop, cost[:,4], 'k--', label='nh=4')
plt.plot(iloop, cost[:,5], 'b--', label='nh=5')
plt.plot(iloop, cost[:,6], 'r--', label='nh=6')

plt.legend()    
plt.xlabel('iteration')
plt.ylabel('cost')

plt.tight_layout(h_pad=0.5, w_pad=1)
plt.savefig('cost.eps', format='eps', dpi=1000)
plt.show()

