# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:39:19 2024

@author: xin

"""

import numpy as np
from scipy.spatial.distance import cdist
import scipy
import warnings
from scipy.linalg import solve_triangular
from scipy.linalg import  cho_solve
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import time
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
        x3 = xn_1+ (a * xn_1 + b * xn_2 ) * dt + \
            (c0 +c * xn_1 + d * xn_2) * np.random.normal(scale=np.sqrt(dt), 
                                                         size = (K))
        
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

class kFoldCV:
    '''
    This class is to perform k-Fold Cross validation on a given dataset
    '''
    def __init__(self, K, numFolds = 10):
        '''
        

        Parameters
        ----------
        K : int
            number of paths.
        numFolds : int, optional
            number of bagging. The default is 10.

        Returns
        -------
        None.

        '''
        self.K = K
        self.numFolds= numFolds
        self.folds = self.CVindexSplit(self.K, self.numFolds)

    def CVindexSplit(self, K, numFolds):
        '''
        Description:
            Function to split the data into number of folds specified
        Input:
            K: number pf paths
            numFolds: integer - number of folds into which the data is to be split
        Output:
            indexSplit, indices of data in each fold
        '''

        indexSplit =[]
        self.foldSize = int(K / numFolds) #foldSize, how many samples in each fold

        index = np.split(np.arange(K), self.numFolds)
        for idx in index:
            indexSplit.append(idx)
        return indexSplit
       

def SliceX(X, t,win_size = 1):
    if t+1 - win_size >= 0:
        return X[:, t+1- win_size: t+1,0]
    else:
        return X[:, :t+1,0]

def orderingsum(x,y):
    result = np.zeros(len(x))
    # for i in range(len(x)):
    #     idx =( x[:,0]<= x[i,0]) & (x[:,1]<= x[i,1] )
    #     #print(y[idx])
    #     result[i] = np.sum(y[idx])
    for i in range(len(x)):
        idx = (x<=x[i]).all(1)
        result[i] = np.sum(y[idx])
    '''    
    # compare x to itself and aggregate with all
    idx = (x <= x[:, None]).all(axis=2)

    # broadcast y using the boolean mask
    # to have the value if True and 0 if False
    # then sum
    result = (y * idx).sum(axis=1)
    '''
    return result

#%%
class GPR:
    def __init__(self, len_scale = 0.45, h= 1 ,noise = 1,normalize_y = False):
        # parameters of the kernel function
        self.len_scale = len_scale  
        self.noise = noise   
        self.h = h
        self.normalize_y = normalize_y 
    def kernel(self, x1, x2):

        dists = cdist(x1 , x2 , metric="sqeuclidean")
        K = self.h**2* np.exp(- (1/(2* self.len_scale**2)) * dists)
        return K

    def fit(self,X_train, y_train):
        self.X_train = X_train  # supplied by learn()
        self.y_train = y_train
        
        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std =np.std(y_train, axis=0) 
  
            # Remove mean and make unit variance
            y_train = (y_train - self._y_train_mean) / self._y_train_std
  
        else:
            shape_y_stats = (y_train.shape[1],) if y_train.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K11 =  self.kernel(self.X_train, self.X_train)
        # print(K11[0,0])
        K11[np.diag_indices_from(K11)] += self.noise
        self.L_ = cholesky(K11, lower=True, check_finite=False)
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve((self.L_, True),self.y_train, check_finite=False)
    def predict(self, X_test, return_cov=False,return_std = False):
  
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )
        
        D = X_test.shape[1]
        
            

        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        K_trans = self.kernel(X_test, self.X_train)
        # print('K_trans:',K_trans[0,0])
        y_mean = (K_trans @ self.alpha_).flatten()
        # undo normalisation
        y_mean = self._y_train_std * y_mean + self._y_train_mean
        
        # y_mean = self._y_train_std * y_mean + self._y_train_mean
        # print('y_mean:',y_mean[0])
        # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
        V = solve_triangular(
            self.L_, K_trans.T, lower=True, check_finite=False)
        self.partialdev_K21 = (-1/(self.len_scale**2))**D *  self.h**(2*D) * \
            ((X_test[:,None]-self.X_train).prod(axis=-1)) * K_trans
        
        partial_diff = (self.partialdev_K21 @ self.alpha_).flatten()
                
        if return_cov:
            # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
            y_cov = self.kernel(X_test,X_test) - V.T @ V
            
            y_cov[np.diag_indices_from(y_cov)] = np.maximum(np.diag(y_cov) ,0 )
            # print('y_cov:',y_cov[0,0])
                # undo normalisation
            y_cov = np.outer(y_cov, self._y_train_std**2).reshape(y_cov.shape, -1)
                # if y_cov has shape (n_samples, n_samples, 1), reshape to
            # (n_samples, n_samples)
            if y_cov.shape[2] == 1:
                y_cov = np.squeeze(y_cov, axis=2)
            return y_mean, y_cov, partial_diff
        elif return_std:
            
            y_var = np.diag(self.kernel(X_test, X_test)).copy()
            
            y_var -= np.einsum("ij,ji->i", V.T, V)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. "
                    "Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0
                # y_var = np.outer(y_var, self._y_train_std**2).reshape(*y_var.shape, -1)
                # undo normalisation
            y_var = np.outer(y_var, self._y_train_std**2).reshape(y_var.shape, -1)
                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
            if y_var.shape[1] == 1:
                y_var = np.squeeze(y_var, axis=1)
            return y_mean, np.sqrt(y_var), partial_diff
        else: 
            return y_mean, partial_diff
#%%

T = 1
dt = 1/50
r = 0
N = int(T/dt)
K = 500
K_test = 500

a = 0.5
b = -0.5

c0 = 2
c = 0
d = 0
window_size = 2
initial = [2,2]

len_scale = 0.45
numFolds = 100
noise = 1
h = 1
#%%
# Implement the Delta algorithm
np.random.seed(42)



SDDEobj = SDDEclass(T, N , a , b, c0 , c ,d , K, initial)
S = SDDEobj.simulatepaths()
all_payoff = SDDEobj.dis_PAYOFF(S,  r, K, T, N)

SDDEobj = SDDEclass(T, N , a , b, c0 , c ,d , K_test, initial)
S_test = SDDEobj.simulatepaths()
all_payoff_test = SDDEobj.dis_PAYOFF(S_test,  r, K_test, T, N)

estimators = [None] * N
time_mat_all= np.zeros((K,N+1)).astype(int) #store time at which the path is stopped
time_mat_all[:,-1] = N

time_mat_test= np.zeros((K_test,N+1)).astype(int) #store time at which the path is stopped
time_mat_test[:,-1] = N

V_train = all_payoff[:,-1]
V_test = all_payoff_test[:,-1]

partial_diffmat =  np.zeros((K,N+1))
partial_diff_testmat =  np.zeros((K_test,N+1))
start = time.time()

#%%

for t in reversed(np.arange(1,N)):
    
    X_train = SliceX(S, t,win_size = window_size)
    current_payoff = all_payoff[:,t]
    future_payoff = V_train
    DELTA = (future_payoff - current_payoff) /K
    y_train = orderingsum(X_train,DELTA )
    # t1 = time.time()
    gpr = GPR(len_scale =len_scale, h = h, noise =noise)
    gpr.fit(X_train,y_train)
    y_hat,  partial_diff = gpr.predict(X_train)
    partial_diffmat[:,t] =partial_diff
    decision_matrix = partial_diff < 0 
    # print((decision_matrix==1).sum())
    time_mat_all[:,t] = np.where(decision_matrix== 1, t,time_mat_all[:,t+1] )
    # V_train =  all_payoff[np.arange(K),time_mat_all[:,t]]

    V_train[decision_matrix] =  current_payoff[decision_matrix] 
    V_train[~decision_matrix] =  (all_payoff[:,t] +1.032* partial_diff )[~decision_matrix] 

    # t2 = time.time()
    # print("time : %3d, training value : %.4f" % (t, V_train.mean()))
    # if t % 5 == 0 or t ==N-1:
    #     fig, ax= plt.subplots(  figsize=(6, 6))
    #     # Plot the distribution of the function  
    #     # ax.scatter(X_train[:,1], y_hat, c='blue', s = 2, label='fitted data')
        
    #     # ax.scatter(X_train[:,1], y_train, c='red', s = 2, label='empirical data')
    #     # ax.scatter(X_train[:,1][decision_matrix], y_hat[decision_matrix], c = 'green', s = 10)
    #     ax.scatter(X_train[:,0][decision_matrix], X_train[:,1][decision_matrix], c = 'green', s = 10)
    #     ax.set_xlabel('$x_n$')
    #     ax.set_ylabel('$F_n$',rotation = 0)
    #     plt.legend()
    #     plt.title('training data, n =' + str(t))
    
    estimators[t]= gpr
    # start += (t2 - t1)

finish = time.time()
value = max(V_train.mean(), all_payoff[0,0])
time_mat_test[:,0] = np.where(all_payoff[0,0] > V_train.mean(), 0, time_mat_test[:,1] )


#%%
# This is to produce the plots in Chapter 6. Figure 24
# t = 10
# XX = X_train.copy()
# k_sort = np.argsort(XX[:,1])
# arguments = XX[:,1]*0
# a = XX[k_sort[0], 1]
# arguments[0] = a
# for k in range(1, K):
#     a += np.linalg.norm(XX[k_sort[k]] - XX[k_sort[k-1]])
#     arguments[k] = a
    


# start_idx = 150
# fi_idx = 200

# XXX= list()

# for k in range(start_idx, fi_idx):
#     a = np.linspace(arguments[k], arguments[k+1],100)-arguments[k]
#     for b in a:
#         x = XX[k_sort[k]] + b *(XX[k_sort[k+1]] - XX[k_sort[k]])/ (arguments[k+1] - arguments[k])
#         x = np.append(x, [arguments[k] + b])
#         XXX.append(x)
# K_min = len(XXX)
# XXX= np.array(XXX)
# y_estim = gpr.predict(XXX[:,:-1])[0]
# y_esti_blue = gpr.predict(X_train)[0]

# fig, ax= plt.subplots(  figsize=(6, 6));

# ax.plot(arguments[start_idx:fi_idx], y_esti_blue[k_sort[start_idx:fi_idx]], c = 'blue',label='fitted data')
# ax.plot(arguments[start_idx:fi_idx], y_train[k_sort[start_idx:fi_idx]],c = 'red',label='empirical data')
# ax.plot(XXX[:, 1], y_estim, c = 'black', label = 'fine grid fitted data')
# ax.axvline(52.3414, c = 'green')
# ax.axvline(55.2329, c = 'green')
# ax.legend()

# fig, ax= plt.subplots(  figsize=(6, 6));
# ax.plot(X_train[:,1][k_sort[start_idx:fi_idx]], y_esti_blue[k_sort[start_idx:fi_idx]], c = 'blue',label='fitted data')
# ax.plot(X_train[:,1][k_sort[start_idx:fi_idx]], y_train[k_sort[start_idx:fi_idx]],c = 'red',label='empirical data')
# ax.plot(XXX[:, 1], y_estim, c = 'black', label = 'fine grid fitted data')

# ax.axvline(1.746, c = 'green')
# ax.axvline(1.7721, c = 'green')
# ax.axvline(1.6228, c = 'green')
# ax.set_xlabel('$X_n$')
# ax.set_ylabel('$F_n$',rotation = 0)
# plt.legend()

# plt.plot(arguments, y_esti_blue[k_sort], c = 'blue')
# plt.plot(arguments, y_train[k_sort],c = 'red')
#%%
'''
# testing 
for t in reversed(np.arange(1,N)):  
    gpr = estimators[t]
    
    X_test = SliceX(S_test, t,win_size = window_size)
    current_payoff = all_payoff_test[:,t]
    future_payoff = V_test
    DELTA = (future_payoff - current_payoff) /K
    y_hat = orderingsum(X_test,DELTA )
    y_test, partial_diff_test = gpr.predict(X_test)
    partial_diff_testmat[:,t] = partial_diff_test
    decision_matrix_test = partial_diff_test < 0 
    time_mat_test[:,t] = np.where(decision_matrix_test== 1, t,time_mat_test[:,t+1] )
    V_test =  all_payoff_test[np.arange(K_test),time_mat_test[:,t]]
    print("time : %3d, test value : %.4f" % (t, V_test.mean()))
    # if t % 5 == 0 or t ==N-1:
    #     fig, ax= plt.subplots(  figsize=(6, 6))
    #     # Plot the distribution of the function  
    #     # ax.scatter(X_test[:,1], y_hat, c='blue', s = 2, label='fitted data')
        
    #     # ax.scatter(X_test[:,1], y_test, c='red', s = 2, label='empirical data')
    #     # ax.scatter(X_test[:,1][decision_matrix_test], y_hat[decision_matrix_test], c = 'green', s = 10)
    #     ax.scatter(X_test[:,0], X_test[:,1], c = 'blue', s = 10, label= 'empirical data')

    #     ax.scatter(X_test[:,0][decision_matrix_test], X_test[:,1][decision_matrix_test], c = 'red', s = 10, label = 'stopped paths')
    #     ax.set_xlabel('$x_{n-1}$')
    #     # ax.set_ylabel('$F_n$',rotation = 0)
    #     ax.set_ylabel('$x_n$',rotation = 0)
        
    #     plt.legend()
    #     plt.title('test data, n =' + str(t))
value_test = max(V_test.mean(), all_payoff_test[0,0])
time_mat_test[:,0] = np.where(all_payoff_test[0,0] > value_test.mean(), 0, time_mat_test[:,1] )

print('training value %.3f'%(value))
print('test value %.3f'%(value_test))
print('training time %.3f'%(finish - start))
# print('training time %.3f'%(start))
'''

#  This part of code include bagging


# cv = kFoldCV(K , numFolds)

# V_est = all_payoff[:,-1]
# for t in reversed(np.arange(1,N)):
#     print(t)
#     X_train = SliceX(S, t,win_size = window_size)
#     decision_matrix = np.zeros((K)).astype(int)
#     # print('t=',t )
#     current_payoff = all_payoff[:,t]
#     future_payoff = V_est
#     DELTA = (future_payoff - current_payoff) /K
#     estimator_t=[None]*numFolds

#     #cross-validation
#     for i in range(numFolds):
#         y_train = orderingsum(X_train[cv.folds[i],:], DELTA[cv.folds[i]])
#         gpr = GPR(len_scale =len_scale, h = h, noise =noise)
        
#         trainset = X_train[cv.folds[i],:]

#         gpr.fit(X_train[cv.folds[i],:],y_train)
#         estimator_t[i] = gpr 
#     estimators[t] = estimator_t
    
#     for i in range(numFolds):
#         bags = list(range(numFolds))
#         bags.remove(i)
#         for bag in bags:
#             model =estimator_t[bag]
#             testset = X_train[cv.folds[i],:]
            
#             _, partial_diff=model.predict(testset)
#             decision_matrix[cv.folds[i]] += partial_diff < 0

#     decision_matrix = decision_matrix > ((numFolds-1)/2)


#     time_mat_all[:,t] = np.where(decision_matrix== 1, t,time_mat_all[:,t+1] )
#     V_est =  all_payoff[np.arange(K),time_mat_all[:,t]]
#     # print(V_est.mean())
#     decision_matrix_test = np.zeros((K_test)).astype(int)
#     X_test = SliceX(S_test, t,win_size = window_size)
#     current_payoff = all_payoff_test[:,t]
#     future_payoff = V_test    
#     DELTA = (future_payoff - current_payoff) /K
#     for i in range(numFolds):
#         for bag in estimators[t]:
#             testset = X_test[cv.folds[i],:]
            
#             _,  partial_diff_test = bag.predict(testset)
#             decision_matrix_test[cv.folds[i]] += partial_diff_test < 0
#     decision_matrix_test = decision_matrix_test  >  (numFolds)/2
#     time_mat_test[:,t] = np.where(decision_matrix_test == 1,t,time_mat_test[:,t+1])
#     V_est_test =(all_payoff_test[np.arange(K_test),time_mat_test[:,t]])
    
# finish  =   time.time()
    
# value = max(V_est.mean(), all_payoff[0,0])
# time_mat_all[:,0] = np.where(all_payoff[0,0] > V_est.mean(), 0, time_mat_all[:,1] )

# V_est_test= np.zeros(K_test)
# V_est_test =all_payoff_test[:,N]
# time_mat_test= np.zeros((K_test,N+1)).astype(int)
# time_mat_test[:,-1]  = N
    
# for t in reversed(np.arange(1,N)):  
#     decision_matrix_test = np.zeros((K_test)).astype(int)
#     X_test = SliceX(S_test, t,win_size = window_size)
#     current_payoff = all_payoff_test[:,t]
#     future_payoff = V_test    
#     DELTA = (future_payoff - current_payoff) /K
#     for i in range(numFolds):
#         for bag in estimators[t]:
#             testset = X_test[cv.folds[i],:]
            
#             _,  partial_diff_test = bag.predict(testset)
#             decision_matrix_test[cv.folds[i]] += partial_diff_test < 0
#     decision_matrix_test = decision_matrix_test  >  (len(estimators[1]))/2
#     time_mat_test[:,t] = np.where(decision_matrix_test == 1,t,time_mat_test[:,t+1])
#     V_est_test =(all_payoff_test[np.arange(K_test),time_mat_test[:,t]])
    
#     # if t % 5 == 0 :
#     #     fig, ax= plt.subplots(  figsize=(6, 6))
#     #     # Plot the distribution of the function  
#     #     ax.scatter(X_test[:,1], y_hat, c='blue', s = 2, label='fitted data')
#     #     ax.scatter(X_test[:,1][decision_matrix_test], y_hat[decision_matrix_test], c = 'green', s = 10)
#     #     ax.scatter(X_test[:,1], y_test, c='red', s = 2, label='empirical data')
    #     plt.legend()
    #     plt.title('test ' +str(t))
# V_est_test = max(V_est_test.mean(), all_payoff_test[0,0])
# time_mat_test[:,0] = np.where(all_payoff_test[0,0] > V_est_test.mean(), 0, time_mat_test[:,1] )

# print('training value %.3f'%(value))
# print('test value %.3f'%(V_est_test))
# print('training time %.3f'%(finish - start))



#%%

'''
fig, ax= plt.subplots(  figsize=(6, 6))
# Plot the distribution of the function  
# ax.scatter(X_train[:,1], y_hat, c='blue', s = 2, label='fitted data')
ax.scatter(X_train[:,1], y_train, c='red', s = 2, label='empirical data')
'''


