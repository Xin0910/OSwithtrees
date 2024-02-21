#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:39:42 2021

@author: 41546
Simulation is based on the paper 'interpretable optimal stopping'
"""

from Deltaalgo import  trainingDelta, testDelta
import numpy as np
import time
import pandas as pd
#%%
class stock:
    def __init__(self, T, C, sigma, So, r, N, K, d, delta=0, symmetric = True):
        self.T = T                   #terminal time
        self.C=C                     #strike price 
        
        self.delta=delta
        self.So=So                   #initial price
        self.r=r                     #riskless interest rate   
        self.N=N                     #number of exercise oppotunity 
        self.K=K                     #number of paths
        self.d=d                     #number of assets
        self.symmetric = symmetric
        self.sigma=self.cal_sigma(sigma)
        self.dt=self.T/self.N
    
    def cal_sigma(self, sigma):
        if self.symmetric == True:
            return sigma
        else:
            if self.d > 5:
                return 0.1+np.arange(1,self.d+1)/(2*self.d)
            else:
                return 0.08+0.32*(np.arange(1,self.d+1)-1)/(self.d-1)
            
    def simulatepaths(self):
        
        S = np.zeros((self.K, self.N + 1, self.d),dtype= np.float32)
        S[:,0,:] = self.So        
        Z=np.random.standard_normal((self.K, self.N , self.d)) .astype(np.float32)

        S[:,1:,]=self.So*np.exp(np.cumsum((self.r-self.delta-0.5*self.sigma**2)*self.dt+self.sigma*np.sqrt(self.dt)*Z, axis=1))
        
        return S
    
    def dis_payoff(self,X):
        B= 170
        maxprice = np.amax(X,axis = 2)
        KOind = np.cumprod(maxprice<B,axis=1)#y_t  
        PAYOFF = np.maximum(0,maxprice-self.C)*KOind #g_t
        timematrix=np.ones((self.K,self.N+1))*np.arange(0,self.N+1,1)
        return np.exp(-self.r*(self.T/(self.N+1))*timematrix)*PAYOFF.astype(np.float32)

        
#%%

T = 3
C = 100
sigma=0.2
r = 0.05

So_s = [90,100,110]
d_s=[4,8,16]
K = 100000
K_test = 100000


reptimes = 1
symmetric = True

min_node_size =10
depth =10
eps = 0

N = 53
delta= 0 
numFolds= 10

foldsize=int(K/numFolds)
columnname =['dimension','S_0','value','Literature','SD','t_test']

value=pd.DataFrame(columns=columnname)
value['dimension'] = np.repeat(d_s,len(So_s))
value['S_0'] = So_s *len(d_s) 

features = '4features'
v = np.zeros((len(d_s), len(So_s),reptimes))
comptime_test = np.zeros((len(d_s), len(So_s),reptimes))
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}


for i_d, d in enumerate(d_s):
    print(d)
    for i_s, So in enumerate(So_s):
        print(So)
        for reptime in range(reptimes):            
            
            S = stock(T, C,sigma ,So , r, N, K, d, delta,symmetric = symmetric)   


            np.random.seed(d + So +reptime)
            paths=S.simulatepaths().astype(np.float32)
            all_payoff = S.dis_payoff(paths)
            
            start_train=time.time()
            
            time_mat_all , V_est ,estimators = trainingDelta(paths, N, K, numFolds,all_payoff, features = features, **kwargs)            
            
            
            finish_train=time.time()

            # testing
            
            S = stock(T, C,sigma ,So , r, N, K_test, d, delta,symmetric = symmetric)   
            

            np.random.seed(42 + reptime)
            paths_test=S.simulatepaths().astype(np.float32)
            all_payoff_test = S.dis_payoff(paths_test)
            
            start_test = time.time()
            time_mat_test, V_est_test = testDelta(paths_test, N, K_test,
                                                  estimators,
                                                  all_payoff_test,
                                                  features = features)     
          
            finish_test=time.time()
            print(V_est_test)
            v[i_d, i_s, reptime] = V_est_test
            comptime_test[i_d, i_s,reptime]=round(finish_test-start_test,1)        
            
        value.loc[(value.dimension==d)&(value.S_0==So),'value']=np.mean(v[i_d, i_s,:])
        value.loc[(value.dimension==d)&(value.S_0==So),'SD']=np.std(v[i_d, i_s,:])
        value.loc[(value.dimension==d)&(value.S_0==So),'t_test']=np.mean(comptime_test[i_d, i_s,:])
    
    