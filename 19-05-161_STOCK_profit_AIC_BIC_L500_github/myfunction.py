##========================================================================================
import numpy as np
from scipy import linalg
#import matplotlib.pyplot as plt

#=============================================================================== 
# June 29.2018: redefine sign function
# note: np.sign(0) = 0 but here sign(0) = 1 
def sign(x): return 1. if x >= 0 else -1.

#=============================================================================== 
def gen_binary(w,h0,l):
    n = np.shape(w)[0]
    s = np.ones((l,n))
    for t in range(1,l-1):
        h = h0 + np.sum(w[:,:]*s[t,:],axis=1) # Wij from j to i
        p = 1/(1+np.exp(-2*h))
        s[t+1,:]= np.sign(p-np.random.rand(n))
    return s
#===============================================================================
# June 29.2018: 
# input: time series s 
# output: interaction w, local field h0
#===============================================================================
def fit_interaction(s,nloop):
    l,n= np.shape(s)
    m = np.mean(s,axis=0)
    ds = s - m
    st1 = s[1:]
    
    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.inv(c)
    dst = ds[:-1].T
    H = st1
    W = np.empty((n,n)) ; H0 = np.empty(n)
    
    for i0 in range(n):
        s1=st1[:,i0]
        h = H[:,i0] ; cost = np.full(nloop,100.) ; h0 = 0.
        for iloop in range(nloop):
            h_av = np.mean(h)
            hs_av = np.matmul(dst,h-h_av)/l
            w = np.matmul(hs_av,c_inv)
            h0=h_av-np.sum(w*m)
            h = np.matmul(s[:-1,:],w[:]) + h0
            
            s_model = np.tanh(h)
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
            
            #MSE = np.mean((w[:]-W0[i0,:])**2)
            #slope = np.sum(W0[i0,:]*w[:])/np.sum(W0[i0,:]**2)
            #print(i0,iloop,cost[iloop]) #,MSE,slope)
            
            if cost[iloop] >= cost[iloop-1]:
                break
            
            h = h*s1/s_model
            
        W[i0,:] = w[:]
        H0[i0] = h0
    return W,H0 

#===============================================================================
# June 29.2018: update hidden spin s based on observed s and interaction w
# input: time series s, interaction w, local field h0, number of observed n 
# output: update s
#===============================================================================
#--------------------------------------------------------------------------------
def update_hidden(s,w,h0,n):
    l,n2=np.shape(s)

    h1=np.empty(n2); p11=np.empty(n2); p12=np.empty(n2)
    h2=np.empty((n2, n2))
    
    # t=0:
    t = 0
    for i in range(n,n2):
        s[t,i]=1.
        h2[:,i]=h0[:]+np.sum(w[:,0:n2]*s[t,0:n2],axis=1)
        p1=1/np.prod(1+np.exp(-2*s[t+1,:]*h2[:,i]))
        p2=1/np.prod(1+np.exp(-2*s[t+1,:]*(h2[:,i]-2*w[:,i])))
        s[t,i]=sign(p1/(p1+p2)-np.random.rand())

    # update s_hidden(t): t = 1 --> l-2:
    for t in range(1,l-1):
        # P(S_hidden(t)):
        h1[n:n2]=h0[n:n2]+np.sum(w[n:n2, :]*s[t-1, :], axis=1)
        p11[n:n2]=1/(1+np.exp(-2*h1[n:n2])) # p(s =+1)
        p12[n:n2]=1-p11[n:n2]                # p(s=-1)

        # P(S(t+1)):
        for i in range(n,n2):
            s[t,i]=1.
            h2[:,i]=h0[:]+np.sum(w[:,0:n2]*s[t,0:n2],axis=1)
            p1=p11[i]/np.prod(1+np.exp(-2*s[t+1,:]*h2[:,i]))
            p2=p12[i]/np.prod(1+np.exp(-2*s[t+1,:]*(h2[:,i]-2*w[:,i])))
            s[t,i]=sign(p1/(p1+p2)-np.random.rand())
                          
    # update s_hidden(t): t = l-1:
    h1[n:n2]=h0[n:n2]+np.sum(w[n:n2, :]*s[l-2, :], axis=1)
    p11[n:n2]=1/(1+np.exp(-2*h1[n:n2]))     
    s[l-1,n:n2]=np.sign(p11[n:n2]-np.random.rand(n2-n))
        
    return s
#===============================================================================
# June 29.2018: predict interaction
# input: observed s, number of hidden nh, number of repeat update hidden nrepat
# number of iteration to predict w 
# output: cost_obs, W, H0, s
#===============================================================================
def predict_interaction(s,nh,nrepeat,nloop):
    l,n = np.shape(s)
    sh = []
    if nh>0:
        sh = np.sign(np.random.rand(l,nh)-0.5)
        s = np.hstack((s,sh)) 

    cost_obs = np.empty(nrepeat)
    like_obs = np.empty(nrepeat) ; like_all = np.empty(nrepeat)
    for irepeat in range(nrepeat):
        w,h0=fit_interaction(s,nloop)
        if nh>0:
            s = update_hidden(s,w,h0,n)     
        #MSE = ((w0[:n,:n]-w[:n,:n])**2).mean()
        #slope = (w0[:n,:n]*w[:n,:n]).sum()/(w0[:n,:n]**2).sum()
            
        h = h0 + np.matmul(s[:-1,:],w.T)
        cost_obs[irepeat] = np.mean((s[1:,:n] - np.tanh(h[:,:n]))**2)

        #---------------------------------------------------
        # 2018.08.27: log likelihood
        like_obs[irepeat] = -np.mean(np.log(1+np.exp(-2*s[1:,:n]*h[:,:n])))
        like_all[irepeat] = -np.mean(np.log(1+np.exp(-2*s[1:,:]*h[:,:])))

        print(cost_obs[irepeat],cost_obs[irepeat]*(1+float(nh)/n)) #,MSE,slope)
        
    return cost_obs,w,h0,s[:,n:],like_obs,like_all

##========================================================================================
# June 12.2018: cross_cov
# a,b -->  <(a - <a>)(b - <b>)>   
##-------------------------------------------   
def cross_cov(a,b):
   da = a - np.mean(a, axis=0)
   db = b - np.mean(b, axis=0)
   return np.matmul(da.T,db)/a.shape[0]

##========================================================================================
# July 02.2018: generate binary configuration based on hidden configuration:
# w[i,j]: from j to i
##-------------------------------------------
def gen_binary_obs(w,h0,sh):
    n2 = np.shape(w)[0] 
    l,nh = np.shape(sh)
    n = n2 - nh

    s = np.ones((l,n))
    s[0,:] = np.sign(np.random.rand(n)-0.5)
    s = np.hstack((s,sh))

    for t in range(1,l-1):
        h = h0[:n]+np.sum(w[:n,:]*s[t,:],axis=1) # w[i,j] from j to i , fixed sh
        p = 1/(1+np.exp(-2*h))
        s[t+1,:n] = np.sign(p-np.random.rand(n)) # update only observed spin

    return s[:,:n] # return observed spins
##========================================================================================
# July 2.2018: generate binary data
# input: number of spins n , coupling variance g, data length l
# output: time series s
def simulation_data(n,g,l):
    w = np.random.normal(0.0,g/np.sqrt(n),size=(n,n))
    s = gen_binary(w,0.,l)
    return s
