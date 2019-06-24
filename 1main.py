##==============================================================================
import numpy as np
import sys
from scipy import linalg
#import matplotlib.pyplot as plt

from myfunction import predict_interaction

#===============================================================================
np.random.seed(1)
nupdate_hidden0 = 300
niter = 500

# paramaters:
l = sys.argv[1] ; t2 = sys.argv[2]
#print(l,t2)
l = int(l) ; t2 = int(t2)

s0 = np.loadtxt('close_open_binary.txt')

t1 = t2 - l
s = s0[t1:t2]

#print(np.shape(s))
n = s.shape[1]

nh_list = [0,1,2,3,4,5,6,7,8]
cost_nh = np.empty(len(nh_list))
for nh in nh_list:
    print(l,t2,nh)
    ext_name = '%03d_%04d_%02d.dat'%(l,t2,nh)
    if nh == 0:
        nupdate_hidden = 1
    else:
        nupdate_hidden = nupdate_hidden0

    #------------------------
    cost_obs,w,h0,sh,like_obs,like_all = predict_interaction(s,nh,nupdate_hidden,niter)

    #-----------------------
    if nh == 0: 
        cost_obs_av = cost_obs[0]
    else:
        cost_obs_av = np.mean(cost_obs[100:])

    cost_nh[nh] = cost_obs_av*(1+float(nh)/n)

    np.savetxt('cost/cost_%s'%ext_name,zip(cost_obs,cost_obs*(1+float(nh)/n),like_obs,like_all),fmt='% 3.6f')

    np.savetxt('W/w_%s'%ext_name,w,fmt='% 3.8f')
    np.savetxt('W/h0_%s'%ext_name,h0,fmt='% 3.8f')
    np.savetxt('W/sh_%s'%ext_name,sh,fmt='% 3.1f')

    cost_out=open('cost/cost_av_%s'%ext_name,'w')
    cost_out.write("% i % 3.6f % 3.6f \n"%(nh,cost_obs_av,cost_nh[nh]))
    cost_out.close()

# nh corresponds to cost min:
nh = np.argmin(cost_nh)
print('nh:',nh)
nh_out=open('cost/nh_%s'%ext_name,'w')
nh_out.write("% i % 3.6f % 3.6f \n"%(nh,cost_nh[nh]/(1+float(nh)/n),cost_nh[nh]))
nh_out.close()

