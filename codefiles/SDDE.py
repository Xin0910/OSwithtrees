# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:36 2023

@author: xin
"""

from Deltaalgo import trainingDelta, testDelta
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#%%
class SDDEclass:
    def __init__(self, T, N , a , b, c0 , c ,d , K, initial):
        self.T = T                
        self.N = N        
        self.K = K        
        self.a = a
        self.b = b                
        self.c0 = c0
        self.c = c
        self.d = d
        self.initial = initial
        self.dt = self.T/self.N


    def SDDE(self, xn_1, xn_2,dt, a, b, c0, c, d, K):
        x3 = xn_1+ (a * xn_1 + b * xn_2 ) * dt + (c0 +c * xn_1 + d * xn_2) \
            * np.random.normal(scale=np.sqrt(dt), size = (K))
        
        return x3

    def simulatepaths(self):
        res = np.zeros((self.K , self.N + 1, 1),dtype= np.float32)
        res[:,0,0] = self.initial[0]
        res[:,1,0] =self.SDDE(res[:,0,0], self.initial[1], self.dt,self.a,
                            self.b, self.c0, self.c, self.d,self.K)        
        for n in range( 2, self.N + 1):

            xn_2 = res[:,n-2,0]
            res[:,n,0] =self.SDDE(res[:,n-1,0], xn_2, self.dt,self.a,
                                self.b, self.c0, self.c, self.d,self.K)
        return  res
        
    def dis_PAYOFF(self, S, r, K , T, N):

        payoff = np.zeros((K , N + 1))
        payoff[:,0] = (self.initial[0] + self.initial[1])/2

        for i in range(1, N+1):
            payoff[:,i] = (S[:,i-1,0] + S[:,i,0])/2
        if r == 0:
             return payoff
        
        timematrix = np.ones((K, N+1))*np.arange(0,N+1,1)
        return  (np.exp(-r*(T/N)*timematrix)*payoff).astype(np.float32)


       

def SliceX(X, t,win_size = 1):
    if t+1 - win_size >= 0:
        return X[:, t+1- win_size: t+1,0]
    else:
        return X[:, :t+1,0]
  
def generatebasis(S, degree =2):
    '''
    return a polynomial basis array of S
    '''
    poly = PolynomialFeatures(degree = degree )
    X = poly.fit_transform(S)
    return X
  


#%%


T = 1
dt = 1/50
N_tau = 2
r = 0
N = int(T/dt)
K = 200000
K_test = 200000
a = 0.5
b = -0.5
features = 'S'
pathDep = True

c0 = 2
c = 0
d = 0

min_node_size = 10
numFolds= 10
eps = 0
depth = 10
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}

initial = [2,2]

#%%
# Implement the Delta algorithm
np.random.seed(42)



SDDEobj = SDDEclass(T, N , a , b, c0 , c ,d , K, initial)


paths = SDDEobj.simulatepaths()
all_payoff = SDDEobj.dis_PAYOFF(paths,  r, K, T, N)

paths_test = SDDEobj.simulatepaths()
all_payoff_test = SDDEobj.dis_PAYOFF(paths_test,  r, K_test, T, N)


time_mat_all , V_est ,estimators = trainingDelta(paths, N, K, numFolds,
                                                  all_payoff, 
                                                  features = features,
                                                  pathDep= pathDep,
                                                  window_size = N_tau,
                                                  **kwargs)
print('The training value calculated using the Delta method is %.3f.' %V_est)


time_mat_test, value_test = testDelta(paths_test, N, K_test, estimators, 
                                      all_payoff_test, 
                                      features = features,
                                      window_size = N_tau,
                                      pathDep = pathDep)
print('The test value calculated using the Delta method is %.3f.' %value_test)

#%%

# Implement the Longstaff Schwartz method
value_LS  = all_payoff[:,-1]
value2_LS = all_payoff_test[:,-1]
model =[None] *(N+1)
coef = [None] * (N+ 1)
start_ls = time.time()

for t in reversed(range(1,N)):
    basisf = SliceX(paths, t,win_size = N_tau ) 
    # basisf = np.hstack((basisf, all_payoff[:,t].reshape(-1,1)))

    X  = generatebasis(basisf, 2)
    y = value_LS * math.exp(-r * T/N)
    clf = LinearRegression(fit_intercept = False )
    clf.fit(X, y)
    coef[t] = clf.coef_
    C =  clf.predict(X)
    
    model[t] =clf
    value_LS = np.where(all_payoff[:,t] > C, all_payoff[:,t], value_LS * math.exp(-r * T/N))
    
    basisf2 = SliceX(paths_test, t,win_size = N_tau ) 
    # basisf2 = np.hstack((basisf2, all_payoff_test[:,t].reshape(-1,1)))
    
    X2 = generatebasis(basisf2, 2)

    C_test =  clf.predict(X2)
    value2_LS = np.where(all_payoff_test[:,t] > C_test, all_payoff_test[:,t], value2_LS * math.exp(-r * T/N))


finish_ls = time.time()   
value_LS  = np.mean(math.exp(-r * T/N) * value_LS)
value2_LS = np.mean(math.exp(-r * T/N) * value2_LS)

print('The training value calculated using the LS method is %.3f.'
      %max(value_LS.mean(),paths[0,0]))
print('The testing value calculated using the LS method is %.3f.' 
      %max(value2_LS.mean(), paths_test[0,0]))



def plotfeaturepayoff(grad, feature,ax, t):
    idx1  = np.argsort(feature)
    grad_sort1 = grad[idx1]
    grad_sum1 = np.cumsum(grad_sort1)
    
    temp = -math.inf
    grad_max1 = np.zeros(grad.shape)
    for j, i in enumerate(grad_sum1):
        grad_max1[j] = temp = max(i, temp)
        

    
    ax.scatter(feature[idx1], grad_sum1, s = 1,  
               label  = r'$\Delta$' + ' at n = ' + str(t))
    ax.scatter(feature[idx1], grad_max1, s = 1, 
               label ='max '+ r'$\Delta$' + ' at n = ' + str(t))
    


t = 49

grad =  (all_payoff_test[np.arange(K), time_mat_test[:, t + 1]] -
         all_payoff_test[: , t] )/K


fig = plt.figure()
ax = plt.subplot()
plotfeaturepayoff(grad, paths_test[:, t, 0],ax,t)
plotfeaturepayoff(grad, paths_test[:, t-1, 0],ax,t-1)
plt.xlabel('$X_n$')
plt.ylabel(r'$\Delta$',rotation = 0)
plt.legend()
plt.title('n =  %d'%t)
