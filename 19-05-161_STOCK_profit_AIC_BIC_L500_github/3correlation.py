import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt

from myfunction import cross_cov
from myfunction import gen_binary_obs
from myfunction import gen_binary
#=========================================================================================
# paramaters:
#l = sys.argv[1] ; t1 = sys.argv[2] ; nh = sys.argv[3]
l = 501 ; t2 = 1260 ; nh = 0
print(l,t2,nh)
l = int(l) ; t2 = int(t2) ;  nh = int(nh)
t1 = t2-l

#ext_name = '%02d.dat'%(nh)
ext_name = '%03d_%04d_%02d.dat'%(l,t2,nh)

s0 = np.loadtxt('close_open_binary.txt')
s0 = s0[t1:t2]

c0 = cross_cov(s0[1:],s0[:-1])

#=========================================================================================
w = np.loadtxt('W/w_%s'%ext_name)
h0 = np.loadtxt('W/h0_%s'%ext_name)

n2 = np.shape(w)[0]

if nh > 0:
    sh = np.loadtxt('W/sh_%s'%ext_name)

if nh==1:
    sh = sh[:,np.newaxis] # convert from (l,) --> (l,1)
      
n = n2 - nh

nsim = 200
c_isim = np.empty((n,n,nsim))
for isim in range(nsim):
    
    # if nh == 0:
    s = gen_binary(w,h0,l) # hidden configuration is NOT fixed
    #else:    
    #s = gen_binary_obs(w,h0,sh)

    c = cross_cov(s[1:,:n],s[:-1,:n])
    c_isim[:,:,isim] = c
    
# average of all:
c_av = np.mean(c_isim,axis=2)
c_dev = np.std(c_isim,axis=2)

MSE = np.mean((c0 - c_av)**2)
slope = np.sum(c0*c_av)/np.sum(c0**2)

slope2_av, intercept_av, R_av, p_value, std_err = stats.linregress(c0.flatten(),c_av.flatten())

#--------------------------------------
print(nh,MSE,slope,R_av)
R_out=open('C/R_%s'%ext_name,'w')
R_out.write("% i % f % f % f \n"%(nh,MSE,slope,R_av))
R_out.close()

C_out=open('C/C_%s'%ext_name,'w')
for i in range(n):
    for j in range(n):
        C_out.write("% i % i % f % f \n"%(i+1, j+1, c0[i,j], c_av[i,j]))
C_out.close()

plt.plot([-0.2,0.2],[-0.2,0.2])
#plt.title('nh=%02d'%nh)
plt.scatter(c0,c_av)
#plt.show()
