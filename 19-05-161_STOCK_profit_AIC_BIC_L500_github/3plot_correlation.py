import numpy as np
import matplotlib.pyplot as plt

nh=3


#a = np.loadtxt('C/C_%02d.dat'%nh)
#c0 = a[:,2]
#c = a[:,3]

plt.figure(figsize=(20,16))

nh_list = [0,1,2,3,4,5,6,7,8]
for i,nh in enumerate(nh_list):
    plt.subplot(3,3,i+1)
    a = np.loadtxt('C/C_%02d.dat'%nh)
    c0 = a[:,2]
    c = a[:,3]
    plt.title('nh = % 02d'%nh)
    #plt.title('nh = 01')
    plt.plot([-0.2,0.2],[-0.2,0.2],'r')
    plt.plot(c0, c, 'ko')
    plt.xlabel('C0')
    plt.ylabel('C')

plt.tight_layout(h_pad=0.5, w_pad=1)
plt.savefig('C.eps', format='eps', dpi=1000)
plt.show()

#plt.plot([-0.2,0.2],[-0.2,0.2])
#plt.title('nh=%02d'%nh)
#plt.scatter(c0,c_av)
#plt.show()

