#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:39:42 2021

@author: 41546
"""
#%%
import pandas as pd
from boundary import boundarycurve
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
from LSM import LSM
import matplotlib
import numpy as np
from Deltaalgo import trainingDelta, testDelta
import math
np.random.seed(42)
#%%
def singlepath(S0_hat, V0_hat, T, lambda_, rho, K, r,  n, density_infty, laplace,jumpsize):
    '''
    

    Parameters
    ----------
    S0_hat : float
        initial price.
    V0_hat : float
        initial squared volatility.
    T : float
        maturity.
    lambda_ : TYPE
        parameter of levy processes.
    rho : float
        DESCRIPTION.
    K : float
        strike.
    r : float
        interest rate.
    n : int
        number of exercise opportunities.
    density_infty : float
        parameter of levy processes..
    laplace : float
        parameter of levy processes..
    jumpsize : float
        parameter of levy processes..

    Returns
    -------
    V_s : array
        sample of V.
    S_s : array
        Sample of S.

    '''
    i = 0
    
    # step i
    V_s = np.zeros((n+1),dtype = 'float32')
    V_s[0] = V0_hat
    S_s = np.zeros((n+1),dtype = 'float32')
    S_s[0] = S0_hat

    while i < n:
        #step ii
        N_hat = np.random.poisson(lambda_ * density_infty * T/n)
        # N_hat = 1
        
        #step iii
        if N_hat>0:
            #generata J_1 ,....J_N from PHI_hat
            # idx = np.where(N_hat ==0)[0]
            J = np.random.exponential( 1/jumpsize, (N_hat)).astype(np.float32)  
            T_i = np.sort(np.random.uniform(0, T/n, size = ( N_hat)).astype(np.float32))
            summation = np.sum (J * np.exp( - lambda_ *( T/n- T_i)) )
        else:
            J = 0
            summation = 0
        

        V_s[i+1] = V_s[i] * math.exp(- lambda_ * T/n) + summation
        
        # step iv
        if N_hat !=0:

            U_hat = V_s[i] * (1- math.exp(- lambda_ *T/n))/lambda_ + np.sum (J *(1- np.exp(- lambda_*( T/n-T_i)))/lambda_)
        else:
            U_hat = V_s[i] * (1- math.exp(- lambda_ *T/n))/lambda_ 
        # step v
        N_u = np.random.normal(-0.5 * U_hat, math.sqrt(U_hat))
        
        #step vi
        S_s[i+1] = S_s[i] * np.exp(N_u + (r + lambda_ *laplace) * T/n- rho * np.sum(J))
        i += 1
        # print(V_s)
    return V_s, S_s


def simulatepaths(M,S0_hat, V0_hat, T, lambda_, rho, K, r,  n, density_infty, laplace,jumpsize):
    '''
    Returns
    -------
    V : array
        sample of V with size (M, n +1).
    S : array
        Sample of S with size (M, n +1).

    '''
    S = np.zeros((M, n+1),dtype = 'float32')
    V = np.zeros((M, n+1),dtype = 'float32')
    

    for i in range(M):
        V[i], S[i] = singlepath(S0_hat, V0_hat, T, lambda_, rho, K, r,  n, density_infty, laplace,jumpsize)
    return V,S

def dis_payoff(S, K, r, M , T, N):
    payoff = np.maximum(K - S, 0) 
    timematrix = np.ones((M, N+1))*np.arange(0,N+1,1)
    return  np.exp(-r*(T/N)*timematrix)*payoff.astype(np.float32)

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
    
def gridV(grid, V_test, S_test,tau, interval ):
    S_mean =np.zeros((grid, N))
    
    for t in range(1,N):
        for j in range(grid):
            S_jt = S_test[(tau ==t) & (V_test[:,t] >=interval[j]) & (V_test[:,t]  < interval[j+1]), t]
            if len(S_jt) ==0:
                continue
            
            S_mean[j,t] = np.mean(S_jt)
    return S_mean

def averageSboundary(K, tau, S_test, N):
    S_t_mean = np.zeros(N+1)
    S_t_mean[-1] = K
    for t in reversed(range(1,N)):
        idx = np.where(tau == t)[0]
        col_t_idx = tau[idx]
        if len(idx) == 0:
            S_t_mean[t] = S_t_mean[t+1]
        else:
            S_t_mean[t] = S_test[idx, col_t_idx].mean()
    S_t_mean[0] = S_t_mean[1]
    return S_t_mean

def extractboundary(T, N, K, gridsize, maxvol, tau, S_test):

    S_t_mean = averageSboundary(K, tau, S_test, N)
    
    tau_s = np.linspace(0,T,N+1).astype(np.float32)
    best_r = best_vol = 0
    loss_val  = float('inf')
    for i in np.linspace(0.01, 1,gridsize):
        for v in np.linspace(0.01, maxvol,gridsize): 
            boundary_flip = boundarycurve(K, v, i, tau_s, T,N)
            temp = ((boundary_flip - S_t_mean) ** 2).sum()
            if temp < loss_val:
                loss_val = min(loss_val, temp)
                best_r, best_vol = i, v
                best_boundary = boundary_flip
    
    for i in np.linspace(best_r -1/gridsize, best_r + 1/gridsize ,gridsize + 1):
        for v in np.linspace(best_vol- 2/gridsize, best_vol + 2/gridsize, gridsize + 1):
            boundary_flip = boundarycurve(K, v, i, tau_s, T,N)
            temp = ((boundary_flip - S_t_mean) ** 2).sum()
            if temp < loss_val:
                loss_val = min(loss_val, temp)
                best_r, best_vol = i, v
                best_boundary = boundary_flip
    return best_boundary

def boundarytest(S, K , T, N , r, M, boundary):
    boundary_matrix = np.repeat(boundary.reshape(1,-1),M, axis = 0)
    value_true = S <= boundary_matrix
    value_true[:,-1] = True
    stoppingtime = np.argmax(value_true,axis = 1)
    value_fitted = np.exp(-r * stoppingtime *T/N) * np.maximum(K- S[np.arange(M), stoppingtime], 0)
    return value_fitted.mean()

    
#%%
min_node_size = 10
numFolds= 10

So_s = [85,90,100,110]
T_s = [1,2]
Vo_s = [0.04, 0.09]

lambda_ = 1
rho = 1
K = 100
M  = 200000
M_test = 200000



So_s = [85]
T_s = [1]
Vo_s = [0.04]
# M  = 20
# M_test = 20


r =  0.05
density_infty = 50
laplace = 50 * 200 *(1/200 -1/( 200 + rho))
jumpsize = 200
rho = 1
depth =10
eps = 0
features = 'S'
kwargs = {'min_node_size':min_node_size, 'depth': depth ,'eps' : eps}
columnname =['sigma','S_0','terminal','time','value','LS','time_LS', 'valuefitted', 'valuemean']
value=pd.DataFrame(columns=columnname)
value['sigma'] = list(np.repeat(Vo_s,len(So_s))) * len(T_s)
value['S_0'] = So_s * len(T_s) * len(Vo_s)
value['sigma'] = list(np.repeat(Vo_s,len(So_s))) * len(T_s)
value['terminal'] = np.repeat(T_s, len(So_s) *len(Vo_s))
#%%
#training and testing


for i_v, V0_hat in enumerate(Vo_s):
    print(V0_hat)
    for i_s, S0_hat in enumerate(So_s):
        print(S0_hat)
        for i_t, T in enumerate(T_s):
            print(T)
            N = int(50 * T)
            np.random.seed(42)

            V,S = simulatepaths(M,S0_hat, V0_hat, T, lambda_, rho, K, r, N,
                                density_infty, laplace, jumpsize)
            all_payoff = dis_payoff(S, K, r, M , T, N)
            
            paths = np.stack((V,S),axis = 2)
            # paths = np.expand_dims(V, 2)
            # paths = np.expand_dims(S, 2)
            start=time.time()
            time_mat_all , V_est ,estimators = trainingDelta(paths, N, M, numFolds,
                                                              all_payoff, features = features,
                                                              **kwargs)            
            finish=time.time()
            

            
            V_test,S_test = simulatepaths(M_test,S0_hat, V0_hat, T, lambda_, 
                                          rho, K, r,  N, density_infty, 
                                          laplace, jumpsize)

            all_payoff_test = dis_payoff(S_test, K, r, M_test , T, N) 
            paths_test =np.stack((V_test,S_test),axis = 2)
            # paths_test = np.expand_dims(V_test, 2)
            # paths_test = np.expand_dims(S_test, 2)
            start_test = time.time()
            time_mat_test, value_test = testDelta(paths_test, N, M_test,
                                                  estimators,
                                                  all_payoff_test,
                                                  features = features)
            
            finish_test=time.time()
        
            print('The value using test data calculated using our method is %.3f.' %value_test )
            timels = time.time()
            h = np.maximum(K-S, 0)
            h2 = np.maximum(K-S_test, 0)
            valueLS, value2LS, model_LS = LSM(S, S_test, N, T, r,
                                              M, M_test,h, h2, 
                                              features = True,
                                              paths=paths, 
                                              paths_test=paths_test)
            print('The value using test data calculated using LS method is %.3f.' % value2LS )
            timelsf = time.time()
            
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'value']=value_test
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'time']= round(finish- start, 3)
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'LS']= value2LS
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'time_LS']= round(timelsf-timels,3)



            '''
            tau = time_mat_test[:,0]
            idx_tau = np.where((tau!=N) & (tau!=0))[0]
            col_idx = tau[idx_tau]
            V_tau = V_test[idx_tau, col_idx]
            S_tau = S_test[idx_tau, col_idx]
        
        
            #3D plot (t, SV_t, S_t)
            
        
            scatter3D(col_idx, V_tau , S_tau, (0,N), (0, 0.6) , (0, K) , 
                      '$n$' , '$SV_{n}$' , '$S_{n}$')
        
            #projection of 3D plot on (t, S_t)
            
            plot2D(col_idx, S_tau , (0,N), (0,K) , '$n$' , '$S_n$')
            
            
            #Take a slice of the 3D plot. Take n = 49
            n = 49
            idx_n = np.where(tau == n)[0]
        
            plot2D(V_test[idx_n,n],  S_test[idx_n, n] ,
                   (0,0.4), (50,K) , '$n$' , '$S_n$')
            grid = 50
            interval = np.linspace(np.amin(V_test) , np.amax(V_test), grid + 1).reshape(-1,1)
        
            S_mean = gridV(grid, V_test, S_test , tau, interval)
            
            V_j, timestep = np.meshgrid(interval[:grid], np.arange(len(idx_tau)), indexing='ij')
            idx_x, idx_y = np.where(S_mean != 0)[0],np.where(S_mean != 0)[1]
            
            scatter3D(timestep[idx_x, idx_y], V_j[idx_x, idx_y] ,S_mean[idx_x, idx_y], 
                      (0,N), (0,  np.amax(V_test)), (0, K) , 
                      '$n$' , '$SV_n$' , r'$\bar{S}_n$')
            
            
            # #Take a slice of the 3D plot after averaging S_test. Take n = 49
            n = 49
            plot2D( V_j[:, n][S_mean[:, n]!=0], S_mean[:, n][S_mean[:, n]!=0] ,
                    xlim = (0, np.amax(V_test)),
                    ylim = (0, 100) , xlabel ='$SV_n$', ylabel='$S_n$')
        
            
            ts = [10,20,30,40]
            fig = plt.figure()
            ax = fig.add_subplot()
            for n in ts:
                idx_x = np.where(S_mean[:,n]!=0)[0];
                ax.scatter(V_j[:, n][idx_x], S_mean[:, n][idx_x], 
                           s =1,label =r'$n$'+' = %d' % (n) )
            ax.set_ylim(0,100)
            ax.set_xlabel('$SV_n$')
            ax.set_ylabel('$S_n$',rotation = 0)
            ax.legend(loc = 'lower right')
            

            
            # value_mean_fitted 
            best_boundary = extractboundary(T, N, K,  5, 1, tau, S_test)
            value_bboundary = boundarytest(S_test, K , T, N , r, M_test, best_boundary)
            print('The testing value calculated using the best fitted boundary is %.3f.' %value_bboundary)
            
            
            # value_mean S_t_mean  
            S_t_mean = averageSboundary(K, tau, S_test, N)
            value_mean = boundarytest(S_test, K , T, N , r, M_test, S_t_mean)
            print('The testing value calculated using the mean boundary is %.3f.' %value_mean)
            
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'valuefitted']=value_bboundary
            value.loc[(value.sigma==V0_hat)&(value.S_0==S0_hat)&(value.terminal==T),'valuemean']=value_mean
            
            
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(np.arange(N+1),S_t_mean , label = r'$\bar{S}_n$')
            ax.plot(np.arange(N+1),best_boundary , label =  r'$b(n, r^*, \sigma^*)$')
            ax.set_xlabel('$n$')
            ax.set_ylabel(r'$\tilde{S}_n$',rotation = 0)
            ax.legend()
            plt.show()
            
            '''
            
            
