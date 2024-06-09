# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:00:00 2024

@author: xin
"""
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
#%%
class Node:
    def __init__(self, beta = None, threshold= None, left=None,
                 right = None, value =None, depth = None, 
                 deltasum = None, isleaf= False):
        self.beta = beta
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.value = value
        self.isleaf = isleaf
        self.deltasum = deltasum
    def is_leaf(self):
        return self.value is not None

class Deltaalgorithm:
    def __init__(self, min_node_size= 15, factor  =100):
        self.min_node_size = min_node_size
        self.factor = factor
        self.root = None
    def fit(self, X, y):
        self.root = self.growtree(X, y, depth = 0)
        
    def predict(self, X):

        return np.array([self.predict_row(xi, self.root) for xi in X])
    
    def predict_row(self,xi, node):   
        if node.isleaf:
            return node.value

        if xi @ node.beta.T  <= node.threshold:
            return self.predict_row(xi, node.left)
        else:
            return self.predict_row(xi, node.right)
    
    def growtree(self, X, y, depth):
        n_samples, n_features = X.shape
        #check leaf condition
        if n_samples <= self.min_node_size:
            leaf_value ,deltasum= self.weight(y)
            return Node(value =  leaf_value,depth = depth,
                        deltasum = deltasum, isleaf = True)
        maxdiff, beta, splitvalue, y_hat = self.split(X, y)
        if maxdiff <= 0:
            leaf_value, deltasum = self.weight(y)
            return Node(value = leaf_value,depth = depth, 
                        deltasum = deltasum ,isleaf= True)
        lhs = np.where(y_hat <= splitvalue)[0]
        rhs = np.where(y_hat>splitvalue)[0]
        left = self.growtree(X[lhs], y[lhs], depth + 1)
        right = self.growtree(X[rhs], y[rhs], depth + 1)
        return Node(beta = beta, threshold = splitvalue, deltasum = np.sum(y),
                    left = left, right = right, depth = depth)
        
        
    def weight(self,y):
        deltasum = np.sum(y)
        if deltasum < 0:  
            return 1, deltasum
        else: return 0, deltasum
    def split(self, X, y ):  
        d = X.shape[1]
        
        #y is delta_k
        #introduce scaling
        # hyper parameter
        
        sd = np.std(y)
        hyperpara = self.factor * sd

        # linear regression
        reg = LinearRegression().fit(X, y)

        coeff =  reg.coef_.reshape(1,-1)
        
        # need to check if the norm is zero or not
        if np.linalg.norm(coeff) == 0:
            # we have a uniform direction
            beta = np.ones(d) / np.std( d)
        
        else:
            beta_normalised =  coeff / np.linalg.norm(coeff)
            beta = beta_normalised
        
        # candidates of the split points 
        y_hat = X @ beta.T  # Y in Sigurd code

        # order y_hat
        y_idx = np.argsort(y_hat[:, 0])
        y_ordered = y[y_idx]
        F_delta = np.cumsum(y_ordered)  # [y_idx]

        cummax = np.maximum.accumulate(F_delta)
        cummin = np.minimum.accumulate(F_delta)

        Y_ordered = y_hat[y_idx]
        
        max_y = cummax - F_delta
        min_y = F_delta - cummin
        
        maxyidx = np.argmax(max_y)
        max_y_max = max_y[maxyidx]
        
        minyidx = np.argmax(min_y)
        min_y_max = min_y[minyidx]
        # print(min_y)
        # print(hyperpara)
        if max_y[-1] <= hyperpara:
            if max_y_max > hyperpara:
                maxdiff = max_y_max
                splitvalue = Y_ordered[maxyidx,0]
                
            else:
                maxdiff =splitvalue= 0
        elif min_y[-1]  <= hyperpara:
        # elif min_y[-self.controlfac]  <= hyperpara:
            if min_y_max > hyperpara:
                maxdiff = min_y_max
                splitvalue = Y_ordered[minyidx,0]
                
            else:
                maxdiff = splitvalue=0
        else:
            maxdiff = max(max_y_max, min_y_max)
            if max_y_max >= min_y_max:
                splitvalue =  Y_ordered[maxyidx,0]
                
            else:
                splitvalue =  Y_ordered[minyidx,0]
        return maxdiff, beta, splitvalue, y_hat

                
        
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

def trainingnewDelta(paths, N, K, numFolds, all_payoff, features = 'S', 
                  pathDep = False, window_size = 1,**kwargs):
    """

    function to train the Delta algorithm
    
    Parameters
    ----------
    paths : numpy array
        Training data of shape (K, N+1, d)
    N : int
        Number of exercise opportunities.
    K : int
        Number of sample paths.
    numFolds : int
        Number of bagging.
    all_payoff : numpy array
        Discounted payoff of training data with size (K, N+1).
    features : string, optional
        Choose the training data set from 'S', 'SandU', '4features'.
        For more options, customise the function called trainingset.
        The default is 'S'.
    pathDep : string, optional
        Only for paths which consider the past data. 
        The default is False.
    window_size : int
        The size of the features when considering the past data. 
        The default is 1.    

    **kwargs : dict
        Other parameters for the CART tree, 
        {depth = 10, min_node_size = 10,eps = 0}.        
        
    Returns
    ----------
    time_mat_all : numpy array
        Stopping time array with size (K, N + 1).
    value: float
        Fitted value using the training data.
    estimators : list
        Trained model of the algorithm.
    """
    estimators=[None] * N
    cv = kFoldCV(K , numFolds=numFolds)
    time_mat_all= np.zeros((K,N+1)).astype(int) #store time at which the path is stopped
    time_mat_all[:,-1] = N
    V_est = all_payoff[:,-1]
    for t in reversed(np.arange(1,N)):
        decision_matrix = np.zeros((K)).astype(int)
        # print('t=',t )
        X_train = trainingset(paths , all_payoff, t, window_size, features, pathDep)
        current_payoff = all_payoff[:,t]
        future_payoff = V_est
        estimator_t=[None]*numFolds
        if numFolds == 1:
            # no cross-validation
            model =  Deltaalgorithm(**kwargs)  
            model.fit(X_train, (future_payoff - current_payoff)/K)
            estimator_t[0] = model
            estimators[t] = estimator_t
            
            # in-sample testing
            model =estimator_t[0]
            prediction=model.predict(X_train)
            decision_matrix += prediction*1   
            
        else:
            #cross-validation
            for i in range(numFolds):
                model =  Deltaalgorithm(**kwargs)
                trainset = X_train[cv.folds[i],:]
                model.fit(trainset, 
                          (future_payoff[cv.folds[i]]-current_payoff[cv.folds[i]])/(K/numFolds) )
                   
                estimator_t[i] = model 
            estimators[t] = estimator_t
            
            for i in range(numFolds):
                bags = list(range(numFolds))
                bags.remove(i)
                for bag in bags:
                    model =estimator_t[bag]
                    testset = X_train[cv.folds[i],:]
        
                    prediction=model.predict(testset)
                    decision_matrix[cv.folds[i]] += prediction * 1
    
            decision_matrix = decision_matrix > ((numFolds-1)/2)
    
    
        time_mat_all[:,t] = np.where(decision_matrix== 1, t,time_mat_all[:,t+1] )
        V_est =  all_payoff[np.arange(K),time_mat_all[:,t]]
        # print(V_est.mean())
    value = max(V_est.mean(), all_payoff[0,0])
    time_mat_all[:,0] = np.where(all_payoff[0,0] > V_est.mean(), 0, time_mat_all[:,1] )
    return time_mat_all,value, estimators

def testnewDelta(S_test, N, K, estimators, all_payoff_test, features='S', 
              pathDep = False,window_size = 1):
    '''
    
    function to evaluate the values using the testing data.
    
    Parameters
    ----------
    S_test : numpy array
        Test data of shape (K, N+1, d).
    N : int
        Number of exercise opportunities.
    K : int
        Number of sample paths.
    estimators : list
        models generated by trainDelta function.
    all_payoff_test : numpy array
        Discounted payoff of test data with size (K, N+1).
    features : string, optional
        Choose the training data set from 'S', 'SandU', '4features'. 
        For more options, customise the function called trainingset.
        The default is 'S'.   
    pathDep : string, optional
        Only for paths which consider the past data. 
        The default is False.
    window_size : int
        The size of the features when considering the past data. 
        The default is 1.  

    Returns
    -------
    time_mat_test : numpy array
        Stopping time array with size (K, N + 1) with respect to test data.
    value : float
        Value calculated by the Delta algorithm.

    '''

    V_est_test= np.zeros(K)
    V_est_test =all_payoff_test[:,N]
    time_mat_test= np.zeros((K,N+1)).astype(int)
    time_mat_test[:,-1]  = N
    
    for t in reversed(np.arange(1,N)):
        decision_matrix_test = np.zeros((K)).astype(int)
        X_test =  trainingset(S_test, all_payoff_test, t, window_size, features, pathDep)

        for bag in estimators[t]:
            prediction = bag.predict(X_test)
            decision_matrix_test  += prediction * 1
        
        decision_matrix_test =decision_matrix_test  >  (len(estimators[1]))/2
        time_mat_test[:,t] = np.where(decision_matrix_test == 1,t,time_mat_test[:,t+1])
        V_est_test =(all_payoff_test[np.arange(K),time_mat_test[:,t]])
    
    value = np.mean(V_est_test)
    time_mat_test[:,0] = np.where(all_payoff_test[0,0] > value, 0, time_mat_test[:,1] )
    value = max(value, all_payoff_test[0,0])
    return time_mat_test, value
