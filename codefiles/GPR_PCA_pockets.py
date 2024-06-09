# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:23:54 2024

@author: xin
"""


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from fbm import FBM

from scipy.spatial.distance import cdist
import scipy
import warnings
from scipy.linalg import solve_triangular
from scipy.linalg import  cho_solve
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import time
#%%
class FractualBM:
    
    def __init__(self, T,   N, M, d, hurst, initial = 0,  payofftype ='identity'):

        self.T = T                   #terminal time
        self.initial = initial          #initial price
        self.N = N                     #number of exercise oppotunities
        self.M = M                    #number of paths
        self.d = d                     #number of assets
        self.dt=self.T/self.N
        self.hurst = hurst
        self.payofftype = payofftype
        if self.hurst != 1:
            self.fBM  = FBM(n=self.N, hurst=self.hurst, length=self.T, method='cholesky')
    
    def fractualBM1(self):
        """fractional Brownian Motion when hurst paramter H=1"""
        return np.linspace(0, self.T, self.N+1) * np.random.randn(1)
    def simulatepaths(self):
        paths = np.zeros((self.M, self.N + 1, self.d),dtype= np.float32)
        if self.hurst != 1:
            for m in range(self.M):
                for i in range(self.d):
                    paths[m,:,i] = self.fBM.fbm()  + self.initial
        else:
            paths[:,0,:] = self.initial
            paths[:,1:,:] = np.random.normal(scale  =np.sqrt( self.T/ self.N), size = (self.M, self.N, self.d))
            paths =  self.initial + np.cumsum(paths, axis = 1)
        return paths


    def dis_PAYOFF(self, X):
        X= X[:,:,0]
        if self.payofftype == 'identity':
            return X
        elif self.payofftype == 'sin':
            return np.sin(X)
        elif self.payofftype == 'square':
            return X ** 2
        else:
            print('Choose the correct payoff type')

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

class stock:
    def __init__(self, T, C, sigma, So, r, N, K, d, delta=0, symmetric = True):
        """
        

        Parameters
        ----------
        T : int/float
            terminal time.
        C : int/float
            strike price.
        sigma : float
            initial volatility.
        So : int/float
            initial price.
        r : float
            riskless interest rate   .
        N : int
            number of exercise oppotunities.
        K : int
            number of paths.
        d : int
            number of stocks.
        delta : float, optional
            DESCRIPTION. The default is 0.
        symmetric : bool, optional
            set the symmetric feature of the volatility of stocks. 
            The default is True.
        """
        
        self.T = T                   #terminal time
        self.C=C                     #strike price         
        self.delta=delta             #dividend
        self.So=So                   #initial price
        self.r=r                     #riskless interest rate   
        self.N=N                     #number of exercise oppotunities
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
        Z=np.random.standard_normal((self.K, self.N , self.d)).astype(np.float32)

        S[:,1:,]=self.So*np.exp(np.cumsum((self.r-self.delta-0.5*self.sigma**2)*self.dt+self.sigma*np.sqrt(self.dt)*Z, axis=1))
        
        return S
    
    def dis_payoff(self,X):
        """
        

        Parameters
        ----------
        X : array
            paths of stock prices.

        Returns
        -------
        array
            discounted payoff.

        """
        timematrix=np.ones((self.K,self.N+1), dtype = np.float32)*np.arange(0,self.N+1,1, dtype = np.float32)
        max1=np.amax(X-self.C,axis=2)
        return np.exp(-self.r*(self.dt)*timematrix)*np.maximum(max1,0).astype(np.float32)

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
        # res = np.zeros(shape = (X.shape[0],N))
        return X[:, :t+1,0]
        # return res

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

def meandelta(X_train, X_train_pca, DELTA, pockets):
    nb_features = X_train.shape[1]
    Xtrain = np.zeros((pockets[0],pockets[1],nb_features),dtype = np.float32)
    DELTA_mean = np.zeros(shape = (pockets), dtype = np.float32)
    
    
    sort_idx = np.argsort(X_train_pca,axis = 0)
    splitpoint0 = np.linspace(0, len(X_train_pca)-1, pockets[0] + 1, dtype=int)
    splitpoint1 = np.linspace(0, len(X_train_pca)-1, pockets[1] + 1, dtype=int)

    # the median of the principal component with the largest variance, creating the pockets 
    x1 = X_train_pca[sort_idx[splitpoint0,0],0]
    x2 = X_train_pca[sort_idx[splitpoint1,1],1]
    #Per pocket, take the mean of X conditional on the pocket-index-set 
    # and the mean of the deltas conditional on the pocket-index-set. 
    for i in range(pockets[0]):
        for j in range(pockets[1]):
            idx = np.where((x1[i] <= X_train_pca[:,0]) &
                           ( X_train_pca[:,0] <=x1[i+1]) &
                           (x2[j] <= X_train_pca[:,1] ) &
                           ( X_train_pca[:,1] <=x2[j+1]) )[0]
            if len(idx) == 0: continue
            DELTA_mean[i,j] = np.mean(DELTA[idx])
            Xtrain[i,j] = np.mean(X_train[idx],axis = 0)
    DELTA_mean = DELTA_mean.flatten()       
    idx = np.where(DELTA_mean != 0)[0]
    Xtrain = Xtrain.reshape(-1,nb_features)
    
    return Xtrain[idx] , DELTA_mean[idx]


def trainingset(S , all_payoff, t, window_size = 0, features = 'S',  pathDep= False):
    '''
    

    Parameters
    ----------
    S : numpy array
        training data.
    all_payoff : numpy array
        payoff of training data.
    t : int
        current time.
    window_size : TYPE, optional
        DESCRIPTION. The default is 0.
    features : TYPE, optional
        DESCRIPTION. The default is 'S'.
    pathDep : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    numpy array
        return a training dataset.

    '''

    if pathDep == True and features == 'S':
        if t + 1 - window_size > 0:
            return  S[:, t+1- window_size: t+1,0]
        else:
            return S[:, :t+1,0]   
    if pathDep == True and features == 'SandU':
        if t + 1 - window_size > 0:
            X_train   = S[:, t+1- window_size: t+1,0]
        else:
            X_train =  S[:, :t+1,0]   
        return np.hstack((X_train, all_payoff[:,t].reshape(-1,1)))
     
    if features =='S':
        #only S as features

            X_train = S[:,t,:]
    elif features == 'SandU':
        #features are (S, U)
        X_train = np.hstack(( S[:,t,:], all_payoff[:,t].reshape(-1,1)))
    elif features =='4features':
        paths= np.sort(S[:,t,:])
        maxprice = np.amax(paths,axis=1).reshape(-1,1) 
        maxprice2 = paths[:,-2].copy().reshape(-1,1) 
        X_train = np.hstack((maxprice, maxprice2,
                             all_payoff[:,t].reshape(-1,1),
                             maxprice- maxprice2))     
    return X_train
        
  

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
        # self.L_ = cholesky(K11)
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve((self.L_, True),self.y_train, check_finite=False)
    def predict(self, X_test, return_cov=False,return_std = False):
  
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )
        
        D = X_test.shape[1]

        # undo normalisation
        
            
            
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        K_trans = self.kernel(X_test, self.X_train)
        # print('K_trans:',K_trans[0,0])
        y_mean = (K_trans @ self.alpha_).flatten()
        
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

# T = 3
# C = 100
# sigma=0.2
# r = 0.05
# S0 = 90
# d = 2
# N = 9
# delta=0.1
# symmetric = True

T = 1
d = 1
N = 100
hurst = 0.05
window_size = N
initial = 0

T = 1
# dt = 1/50
r = 0
# N = int(T/dt)
K = 5000*4
K_test = 5000*4

pockets =  [16,4]
a = 0.5
b = -0.5

c0 = 2
c = 0
# d = 0
# window_size = 2
# initial = [2,2]

len_scale = 0.45
numFolds = 10
noise =1
h = 1
#%%
# Implement the Delta algorithm
np.random.seed(42)



# SDDEobj = SDDEclass(T, N , a , b, c0 , c ,d , K, initial)
# paths = SDDEobj.simulatepaths()
# all_payoff = SDDEobj.dis_PAYOFF(paths,  r, K, T, N)
# SDDEobj = SDDEclass(T, N , a , b, c0 , c ,d , K_test, initial)
# paths_test = SDDEobj.simulatepaths()
# all_payoff_test = SDDEobj.dis_PAYOFF(paths_test,  r, K_test, T, N)

# S = stock(T, C,sigma ,S0 , r, N, K, d, delta,symmetric = symmetric)   
# paths=S.simulatepaths().astype(np.float32)
# all_payoff = S.dis_payoff(paths)
# S = stock(T, C,sigma ,S0 , r, N, K_test, d, delta,symmetric = symmetric)   
# paths_test=S.simulatepaths().astype(np.float32)
# all_payoff_test = S.dis_payoff(paths_test)
            
S = FractualBM(T,  N, K, d, hurst,initial, payofftype ='identity')     
paths = S.simulatepaths()
all_payoff = S.dis_PAYOFF(paths)
S = FractualBM(T,  N, K_test, d, hurst,initial, payofftype ='identity')     
paths_test = S.simulatepaths()
all_payoff_test = S.dis_PAYOFF(paths_test)
 
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
# t = N-1
    X_train = SliceX(paths, t,win_size = window_size)
    # X_train = trainingset(paths , all_payoff, t, window_size = 0, features = 'SandU',  pathDep= False)
    current_payoff = all_payoff[:,t]
    future_payoff = all_payoff[np.arange(K), time_mat_all[:, t+1]]
    DELTA = (future_payoff - current_payoff) /K
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    pca = PCA(n_components=2)
    pca.fit(X_train_scaler)
    X_train_pca = pca.transform(X_train_scaler)
    Xtrain, y_train = meandelta(X_train_scaler, X_train_pca, DELTA, pockets)
    # zz = np.dot(Xtrain, pca.components_)

    # X = scaler.inverse_transform(zz)  
    gpr = GPR(len_scale =len_scale, h = h, noise =noise)
    gpr.fit(Xtrain,y_train)
    y_hat,  _ = gpr.predict(X_train_scaler)
    
    decision_matrix = y_hat < 0 
    # print((decision_matrix==1).sum())
    time_mat_all[:,t] = np.where(decision_matrix== 1, t,time_mat_all[:,t+1] )
    V_train =  all_payoff[np.arange(K),time_mat_all[:,t]]
    # print("time : %3d, training value : %.4f" % (t, V_train.mean()))
    
    X_test = SliceX(paths_test,t,win_size = window_size)
    # X_test = SliceX(S_test, t,win_size = window_size)
    # X_train = trainingset(paths , all_payoff, t, window_size = 0, features = 'SandU',  pathDep= False)
    # X_test = trainingset(paths_test , all_payoff_test, t, window_size = 0, features = 'SandU',  pathDep= False)
    current_payoff = all_payoff_test[:,t]
    future_payoff = all_payoff_test[np.arange(K_test),time_mat_test[:,t+1]]
    DELTA = (future_payoff - current_payoff) /K

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    X_test_scaler = scaler.transform(X_test)
    y_test, _ = gpr.predict(X_test_scaler)
    # partial_diff_testmat[:,t] = partial_diff_test
    decision_matrix_test = y_test < 0 
    time_mat_test[:,t] = np.where(decision_matrix_test== 1, t,time_mat_test[:,t+1] )
    V_test =  all_payoff_test[np.arange(K_test),time_mat_test[:,t]]
    # print("time : %3d, test value : %.4f" % (t, V_test.mean()))
    # if t % 5 == 0 or t ==N-1:
    #     fig, ax= plt.subplots(  figsize=(6, 6))
    #     # Plot the distribution of the function  
    #     ax.scatter(X_train[:,1], y_hat, c='blue', s = 2, label='fitted data')
    
    #     ax.scatter(X_train[:,1], y_train, c='red', s = 2, label='empirical data')
    #     # ax.scatter(X_train[:,1][decision_matrix], y_hat[decision_matrix], c = 'green', s = 10)
    #     ax.scatter(X_train[:,1][decision_matrix], X_train[:,0][decision_matrix], c = 'green', s = 10)
    #     plt.legend()
    #     plt.title('train ' +str(t))
    
    # estimators[t]= gpr


value = max(V_train.mean(), all_payoff[0,0])
time_mat_all[:,0] = np.where(all_payoff[0,0] > V_train.mean(), 0, time_mat_all[:,1] )


value_test = max(V_test.mean(), all_payoff_test[0,0])
time_mat_test[:,0] = np.where(all_payoff_test[0,0] > value_test.mean(), 0, time_mat_test[:,1] )

print('training value %.3f'%(value))
print('test value %.3f'%(value_test))



#%%

'''
fig, ax= plt.subplots(  figsize=(6, 6))
# Plot the distribution of the function  
# ax.scatter(X_train[:,1], y_hat, c='blue', s = 2, label='fitted data')
ax.scatter(X_train[:,1], y_train, c='red', s = 2, label='empirical data')
'''

