
import numpy as np

class Node:
    '''
    node object. Delta algorithm
    x:         the training data
    gradient:  difference between payoff at time t and payoff at t-1, divided
               by number of samples in the current node
    I:         used to keep track of samples within the tree structure
               an array consisted of indices in one node
    depth:     limits the number of depth in the tree
    min_node_size:  minimum number of samples for a node to be considered a node 
               (complexity parameter)
    initial:    if initial =0, then at root node, we random select one feature
    row_count: number of total samples
    eps:       limits the maximum 
               (complexity parameter)
    column_subsample: indices of all features
    delta:     sum of gradients in the current node
    label:     value of each leaf, 1 or 0
    score:     score of the current node
    small_node_size: check if the number of samples reach the limit
    '''
    def __init__(self, x, gradient, I, min_node_size = 10,
                 depth = 10, eps = 0, initial = 1 ):
      
        self.x = x 
        self.gradient = gradient
        self.I = I 
        self.depth = depth
        self.min_node_size = min_node_size
        self.initial = initial   # zero for initial root, then it swaps to one
        self.row_count = len(I)
        self.eps = eps  
        self.column_subsample =np.arange(x.shape[1])
        self.delta= np.sum(self.gradient[self.I])
        self.label = self.compute_weight(self.delta)
        self.score = 0
        self.small_node_size = False


        if self.initial > 0:
            self.find_varsplit()      
        else: 
            self.col_idx =np.random.randint(0, len(self.column_subsample) -1)
            self.split = self.x[ np.random.randint(0,self.row_count-1) , self.col_idx ]
            x = self.x[self.I, self.col_idx]
            lhs = np.nonzero(x<=self.split)[0]
            rhs = np.nonzero(x>self.split)[0]
            self.is_leaf = self.is_leaf()
            if self.is_leaf: return    #this gives False for is_leaf at root
            self.lhs = Node( x = self.x, gradient = self.gradient[lhs], I = self.I[lhs],
                            min_node_size = self.min_node_size, initial = 1 )
            self.rhs = Node( x = self.x, gradient = self.gradient[rhs], I = self.I[rhs], 
                            min_node_size = self.min_node_size, initial = 1 )

    def compute_weight(self, delta):
        """

        Parameters
        ----------
        delta : float
            sum of gradients in the current node.

        Returns
        -------
        1/0.

        """
        
        if np.sum(delta)< 0: return 1
        else: return 0  
          
    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tree 
        structure.
        If no split is better than the score initalised at the begining 
        then no splits further splits are made
        '''
        for c in self.column_subsample :
            self.find_greedy_split(c)
        
        x = self.split_col
        if type(x)!= str:
            lhs=np.nonzero(x<=self.split)[0]
            rhs=np.nonzero(x>self.split)[0]
            if min(len(lhs) , len(rhs)) < self.min_node_size:
                self.small_node_size  = True
            self.is_leaf = self.is_leaf()
            if self.is_leaf:
                return 
            
            self.lhs=Node(x = self.x, gradient = self.gradient, 
                          I =self.I[lhs], min_node_size = self.min_node_size, 
                          depth = self.depth-1, eps = self.eps )
            self.rhs = Node(x = self.x, gradient = self.gradient,
                            I =self.I[rhs], min_node_size = self.min_node_size, 
                            depth = self.depth-1, eps = self.eps)
        else:
            self.is_leaf = self.is_leaf()
    def find_greedy_split(self,var_idx):
        '''
        for a given feature, we calculate score at each split
        and update the best score and split point
        '''
        x=self.x[self.I,var_idx]
        I_sort=np.argsort(x)
        left_gain = 0
        gradient=self.gradient[self.I]
        for k in range(len(I_sort) -1):
            
            left_gain+=gradient[I_sort[k]]
            right_gain=self.delta-left_gain
            current_score = max(self.score,abs(left_gain),abs(right_gain))
            
            if current_score > self.score *(1+self.eps) :
                self.left_gain = left_gain
                self.right_gain = right_gain
                self.var_idx = var_idx
                self.score = current_score
                self.split = x[I_sort[k]]    
                self.I_sort = I_sort
                self.col_idx = I_sort[k]

    @property
    def split_col(self):
        '''
        extract one column of training dataset
        '''
        try:
            return self.x[self.I,self.var_idx]
        
        except AttributeError:
            return 'leaf'
    
    def is_leaf(self):
        '''
        check if a node is a leaf
        '''

        x=self.split_col
        return   self.score <= abs(self.delta) or self.small_node_size or self.depth <= 0 or type(x)==str

        
    def predict(self,x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self,xi):
        if self.is_leaf:
            return self.label
        if xi[self.var_idx] <= self.split:
            node = self.lhs
        else:
            node = self.rhs
        return node.predict_row(xi)


class BoostClassifier:
    def fit(self, X, current_payoff, future_payoff, depth = 10, 
            min_node_size = 10,   eps = 0, 
            initial=1):
        '''
        fitting the Delta algorithm
        Parameters
        ----------
        X : array
            training data.
        current_payoff : array
            payoff at current time t.
        future_payoff : array
            payoff at time t+1.
        depth : int, optional
            depth of the tree. The default is 10.
        min_node_size : int, optional
            minimum sample size in each node. The default is 10.
        eps : float, optional
            controlling parameters for the degree of improvement. 
            The default is 0.
        initial : int, optional
            Weather randomly choose a splitting feature. The default is 1.

        Returns
        -------
        None.

        '''
        self.X = X
        self.depth = depth
        self.eps = eps
        self.min_node_size = min_node_size
        self.current_payoff = current_payoff
        self.future_payoff = future_payoff    
        self.I=np.arange(len(X))
        self.Grad=(future_payoff-current_payoff)/len(X)
        self.initial = initial
        self.boosting_tree = Node(
            self.X, self.Grad, self.I, min_node_size = self.min_node_size,
            depth = self.depth, eps = self.eps,
            initial= self.initial)
        self.base_pred=self.boosting_tree.predict(self.X)      

    def predict(self, X):
        '''
        

        Parameters
        ----------
        X : array
            test data.

        Returns
        -------
        pred : array
            stopping decisions for each test data.

        '''
        pred = np.zeros((len(X)))
        pred = self.boosting_tree.predict(X)        
        return pred

class TreeDepth:
    '''
    Calculate the maximum depth of a tree.
    Return: int
        maximum depth of a tree.
    '''
    def maxDepth(self, root):
        if root.is_leaf == True:
            return 0
        leftheight = self.maxDepth(root.lhs)
        rightheight = self.maxDepth(root.rhs)
        return max(leftheight, rightheight ) +1

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
        
    
def trainingDelta(paths, N, K, numFolds, all_payoff, features = 'S', 
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
            model = BoostClassifier()
            model.fit(X_train, current_payoff, future_payoff,**kwargs)
            estimator_t[0] = model
            estimators[t] = estimator_t
            
            # in-sample testing
            model =estimator_t[0]
            prediction=model.predict(X_train)
            decision_matrix += prediction*1   
            
        else:
            #cross-validation
            for i in range(numFolds):
                model = BoostClassifier()
                trainset = X_train[cv.folds[i],:]
                model.fit(trainset, current_payoff[cv.folds[i]], 
                          future_payoff[cv.folds[i]], **kwargs)
                   
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

def testDelta(S_test, N, K, estimators, all_payoff_test, features='S', 
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
