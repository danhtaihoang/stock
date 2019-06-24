import numpy as np
import matplotlib.pyplot as plt

#plt.gca().set_color_cycle(['black', 'blue', 'red', 'orange','magenta'])

plt.figure(figsize=(20,16))

a = np.loadtxt('cost_av.dat')
cost_obs = a[:,1]
cost = a[:,2]
nh = a[:,0]

#=========================================================================================
#plt.subplot(3,3,1)
plt.title('cost_av')
plt.plot(nh, cost_obs, 'ko-', label='obs' )
plt.plot(nh, cost, 'r^--', label='entire')

plt.legend()    
plt.xlabel('nh')
plt.ylabel('cost')

plt.tight_layout(h_pad=0.5, w_pad=1)
plt.savefig('cost_av.eps', format='eps', dpi=1000)
plt.show()

