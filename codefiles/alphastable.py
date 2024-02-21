# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:45:50 2023

@author: xin

This file is to create numerical results for alpha-stable processes
"""

import matplotlib.cm as cm
import time
import numpy as np
from scipy.stats import levy_stable
from Deltaalgo import trainingDelta, testDelta
import matplotlib.pyplot as plt

#%%
def laplacef(alpha, beta, rho, mu, scale):
    return  (scale * rho) ** alpha
    
def stable( N , K , d, T , alpha ,beta ,V0, lambda_, r , rho, scale = 1, mu = 0):
    #Euler scheme for simulating alpha-stable processes.
    V = np.zeros((K, N+1))
    V[:,0] = V0 
    dt = T/ N
    # Z is the increment of levy process at each timestep
    Z = levy_stable.rvs(alpha, beta,loc = mu * lambda_ * dt , 
                        scale = scale * (lambda_ * dt)**(1/alpha) , 
                        size=(K,N))

    for t in range(N):
        V[:,t+1] = V[:,t] +  (- lambda_ * V[:,t]) * dt + Z[:,t]
    return V , Z

def simulatepaths( S0, N, K, d, T, alpha, beta, V0, lambda_, r ,rho, scale, mu):
    if beta != 1:
        print('for stable process, please set beta equal to 1 to be a subordinator')
        return 0
    dt = T/N
    S = np.zeros((K, N+1)).astype(np.float32)
    S[:,0] = S0

    V, Z = stable( N , K , d, T , alpha ,beta ,V0, lambda_ , r , rho, scale  , mu  )

    bm = np.random.standard_normal((K,N)) .astype(np.float32)
    laplaceex = laplacef(alpha , beta , rho, mu, scale )
    S[:, 1:] = S0 * np.exp(np.cumsum((r + lambda_ * laplaceex - 0.5 * V[:,1:]) * dt
                                     + np.sqrt(V[:,1:])*np.sqrt(dt) * bm
                                     -rho * Z, axis = 1))
    return V.astype(np.float32), S

def dis_payoff(S, C, r, K , T, N):
    payoff = np.maximum(C - S, 0) 
    timematrix = np.ones((K, N+1))*np.arange(0,N+1,1)
    return  (np.exp(-r*(T/N)*timematrix)*payoff).astype(np.float32)

def gridV(grid, V_test, S_test,tau, interval ):
    S_mean =np.zeros((grid, N))
    
    for t in range(1,N):
        for j in range(grid):
            S_jt = S_test[(tau ==t) & (V_test[:,t] >=interval[j]) & (V_test[:,t]  < interval[j+1]), t]
            if len(S_jt) ==0:
                continue
            
            S_mean[j,t] = np.mean(S_jt)
    return S_mean

def scatter3D(x, y ,z, xlim = None, ylim = None, zlim = None,
              xlabel = None, ylabel = None, zlabel = None):
    
    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    ax.zaxis.set_rotate_label(False)
    ax.scatter(x, y , z , s = 5, c = x, cmap=cm.coolwarm)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation = 0)
    ax.set_zlabel(zlabel, rotation = 0)
    plt.show()

def plot2D(x, y , xlim = None, ylim = None, 
           xlabel = None, ylabel = None):
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y, s = 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation = 0)
    plt.show()
        
    
#%%

alpha = 0.5
beta = 1
T = 1
V0_hat = 0.04
C = 100
K = K_test= 200000
lambda_ = 0.2
S0_hat = 85
r = 0.05
rho = 1
mu = 0
scale = 1
d = 1
min_node_size = 10
numFolds= 10
delta = 0
N =int( 50 * T)
foldsize= int(K / numFolds)
eps = 0
depth = 10
features = 'S'


kwargs = {'min_node_size':min_node_size, 'depth': depth ,'eps' : eps}

#%%

np.random.seed(42)

V,S = simulatepaths( S0_hat, N, K, d, T, alpha, beta,
                    V0_hat, lambda_, r ,rho, scale, mu)

V_test, S_test = simulatepaths( S0_hat, N, K_test, d, T, alpha, beta, 
                               V0_hat, lambda_, r ,rho, scale, mu)



all_payoff = dis_payoff(S, C, r, K , T, N)

paths = np.stack((V,S),axis = 2)
# paths = np.expand_dims(V, 2)
# paths = np.expand_dims(S, 2)

start=time.time()
time_mat_all , V_est ,estimators = trainingDelta(paths, N, K, numFolds,
                                                  all_payoff, 
                                                  features = features,
                                                  **kwargs)            
finish=time.time()




all_payoff_test = dis_payoff(S_test, C, r, K_test , T, N) 

paths_test =np.stack((V_test,S_test),axis = 2)
# paths_test = np.expand_dims(V_test, 2)
# paths_test = np.expand_dims(S_test, 2)

start_test = time.time()
time_mat_test, value_test = testDelta(paths_test, N, K_test,
                                      estimators,
                                      all_payoff_test,
                                      features = features  )

finish_test=time.time()

print('The value using test data calculated using our method is %.3f.' %value_test )


            
tau = time_mat_test[:,0]
idx_tau = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_tau]
V_tau = V_test[idx_tau, col_idx]
S_tau = S_test[idx_tau, col_idx]


            
scatter3D(col_idx, V_tau , S_tau, (0,N), (0, 0.6) , (0, C) , 
          '$n$' , '$SV_{n}$' , '$S_{n}$')


n = 49
idx_n = np.where(tau == n)[0]

plot2D(V_test[idx_n, n],  S_test[idx_n, n] ,
       (0,0.4), (50,C) , '$n$' , '$S_n$')


grid = 50
interval =np.linspace(np.amin(V_tau)-0.0001, 10 +0.0001, grid + 1).reshape(-1,1)
# interval =np.linspace(np.amin(V_tau)-0.0001, np.amax(V_tau)+0.0001, grid + 1).reshape(-1,1)

S_mean = gridV(grid, V_test, S_test , tau, interval)

V_j, timestep = np.meshgrid(interval[:grid], np.arange(len(idx_tau)), indexing='ij')
idx_x, idx_y = np.where(S_mean != 0)[0],np.where(S_mean != 0)[1]

scatter3D(timestep[idx_x, idx_y], V_j[idx_x, idx_y] ,S_mean[idx_x, idx_y], 
          (0,N), (0,  np.amax(V_test)), (0, C) , 
          '$n$' , '$SV_n$' , r'$\bar{S}_n$')


t= 49
plt.scatter( V_j[:, t], S_mean[:, t], s = 5)
plt.ylim(0,100)
plt.xlabel('$SV_n$')
plt.ylabel('$S_n$',rotation = 0)

# to plot a few at the same time
ts = [ 10,20,30,40]
fig = plt.figure()
ax = fig.add_subplot()
for t in ts:
    idx_x = np.where(S_mean[:,t]!=0)[0];
    ax.scatter(V_j[:, t][idx_x], S_mean[:, t][idx_x], s = 5,label =r'$\tau^*$'+' = %d' % (t) )
plt.legend(loc = 'upper right')


