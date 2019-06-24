##========================================================================================
##========================================================================================
import numpy as np

n=25
cost_out=open('cost_av.dat','w')
for nh in range(0,9):
    cost = np.loadtxt('cost/cost_%02d.dat'%(nh))

    if nh == 0:
        cost_obs = cost[0]
    else:
        cost_obs = np.mean(cost[100:,0])
    cost_all = cost_obs*(1.+float(nh)/n)

    cost_out.write("% i % f % f \n"%(nh,cost_obs,cost_all))
    print(nh,cost_obs,cost_all)

cost_out.close()  



  
