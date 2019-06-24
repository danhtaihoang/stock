import numpy as np
import sys
#from scipy import stats
import matplotlib.pyplot as plt

#===============================================================================
# paramaters:
l = 501

s0 = np.loadtxt('close_open_binary.txt')
n = np.shape(s0)[1]

ds0 = np.loadtxt('close_open_percent.txt')

#===============================================================================
t2min = 754
t2max = 3394
profit = np.zeros(t2max-t2min)
cost_av = np.zeros(t2max-t2min)
p_max = np.zeros(t2max-t2min)
nh=0
it=0 ; itrade=0
profit2 = np.zeros(t2max-t2min)

t2list = np.arange(t2min,t2max)

nh0 = np.loadtxt('number_hidden_nhmax6.dat').astype(int)  
nhs = nh0[:,1]  # cost
#nhs = nh0[:,2] # AIC
#nhs = nh0[:,3] # BIC

print(len(nhs))

#for t2 in range(t2min,t2max):
for it,t2 in enumerate(t2list):
#for t2 in range(t2min,t2min+1):
    t1 = t2-l 
    s = s0[t1:t2]    
    ds = ds0[t2]
        
    # read nh
    #nh0 = np.loadtxt('cost/nh_%03d_%04d_06.dat'%(l,t2))    
    #nh = int(nh0[0])
    #print('nh:',nh)

    nh = nhs[it]
    
    ext_name = '%03d_%04d_%02d.dat'%(l,t2,nh)      
    w = np.loadtxt('W/w_%s'%ext_name)
    h0 = np.loadtxt('W/h0_%s'%ext_name)
    
    if nh > 0:
        sh = np.loadtxt('W/sh_%s'%ext_name)
        if nh == 1:
            sh = sh[:,np.newaxis]
        s = np.hstack((s,sh))
    
    # at the final day:
    h_final = h0[:n] + np.sum(w[:n,:n]*s[l-1,:n],axis=1)
    p = 1/(1+np.exp(-2*h_final))
    
    # at days in the time window
    h = h0[:n] + np.matmul(s[:l-1,:n],w[:n,:n].T)
    
    cost = np.mean((s[1:,:n]-np.tanh(h[:,:n]))**2,axis=1)
    
    cost_av[it] = np.mean(cost)

    #if (cost[l-2]<=np.mean(cost[:l-2])):
    
    i1 = np.argmax(p)  # stock is predicted to be increased   
    i2 = np.argmin(p) # stock is predicted to be decreased
   
    profit[it] = ds[i1] - ds[i2] # trade every day
    
    if ds0[t2-1,i1] < 0 :    
        profit2[it] = ds[i1]
        itrade += 1
 
    if ds0[t2-1,i2] > 0 :
        profit2[it] = profit2[it] - ds[i2]
        itrade += 1
                
    #it += 1
        
p_cum = np.cumsum(profit)
p2_cum = np.cumsum(profit2)

np.savetxt('profit_cost_nhmax6.dat',zip(p_cum,p2_cum),fmt='% 3.6f')
#np.savetxt('profit_AIC_nhmax6.dat',zip(p_cum,p2_cum),fmt='% 3.6f')
#np.savetxt('profit_BIC_nhmax6.dat',zip(p_cum,p2_cum),fmt='% 3.6f')

#-----------------------
print(it, itrade)

plt.plot(p_cum,'b-')
plt.plot(p2_cum,'r-')

#-----------------------
#plt.show()
profit_out=open('profit_transaction_nh.dat','w')
profit_out.write("% i % i % 3.6f % 3.6f \n"%(2*it,itrade,p_cum[-1]/(2*it),p2_cum[-1]/itrade))
profit_out.close()
