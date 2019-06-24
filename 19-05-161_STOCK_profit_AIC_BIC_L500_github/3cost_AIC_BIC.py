import numpy as np
#import matplotlib.pyplot as plt

n = 25
l = 500
#nhs = [0,1,2,3,4,5,6,7,8]
nhs = [0,1,2,3,4,5,6]
nnh = len(nhs)

nh_out = open('number_hidden_nhmax6.dat','w')

cost = np.empty(nnh) ; AIC = np.empty(nnh) ; BIC = np.empty(nnh)
for t2 in range(754,3395):
    for ih,nh in enumerate(nhs):   
        ext_name = '501_%04d_%02d.dat'%(t2,nh)   
        try:
            f = np.loadtxt('cost/cost_%s'%(ext_name))            
        except:
            print('cannot find this file:',g,nh0,nh,ln)
        
        if nh == 0:
            cost[ih] = f[1]
            #like = l*n*f[2]
            like = l*n*f[3]  
        else:
            cost[ih] = (f[100:,1]).mean()
            #like = l*n*f[:,2].max()   # likelihood of observed variables
            like = l*(n+nh)*f[:,3].max()  # likelihood of entire system

        # Akaike information criterion (AIC)    
        k = (n+nh)**2    

        AIC[ih] = 2*(k - like) 

        # Bayesian information criterion (BIC)
        BIC[ih] = np.log(l)*k - 2*like
               
        #print(nh,cost[ih],int(AIC[ih]),int(BIC[ih]))
        
    nh_out.write("% i % i % i % i \n" %(t2,nhs[int(np.argmin(cost))],
                nhs[int(np.argmin(AIC))],nhs[int(np.argmin(BIC))]))

nh_out.close()
#=========================================================================================
"""
# plot:   
plt.figure(figsize=(11,3.2))
plt.subplot2grid((1,3),(0,0))
plt.title('Discrepancy')
plt.xticks(nhs)    
plt.plot(nhs,cost,'ko-')

plt.subplot2grid((1,3),(0,1))
plt.xticks(nhs)
plt.title('AIC')
plt.plot(nhs,AIC,'ko-')

plt.subplot2grid((1,3),(0,2)) 
plt.xticks(nhs)
plt.title('BIC')
plt.plot(nhs,BIC,'ko-') 

plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.savefig('cost_AIC_BIC_like_entire.pdf', format='pdf', dpi=100)
"""    

        
