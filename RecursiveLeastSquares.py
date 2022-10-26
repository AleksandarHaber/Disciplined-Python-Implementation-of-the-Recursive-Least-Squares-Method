# -*- coding: utf-8 -*-
"""
This is an implementation of the recursive least-squares method that is derived and explained here

https://aleksandarhaber.com/introduction-to-kalman-filter-derivation-of-the-recursive-least-squares-method-with-python-codes/

Author: Aleksandar Haber 
Last Revision: October 25, 2022

"""

class RecursiveLeastSquares(object):
    
    # x0 - initial estimate used to initialize the estimator
    # P0 - initial estimation error covariance matrix
    # R  - covariance matrix of the measurement noise
    def __init__(self,x0,P0,R):
        
        # initialize the values
        self.x0=x0
        self.P0=P0
        self.R=R
        
        # this variable is used to track the current time step k of the estimator 
        # after every time step arrives, this variables increases for one 
        # in this way, we can track the number of variblaes
        self.currentTimeStep=0
                  
        # this list is used to store the estimates xk starting from the initial estimate 
        self.estimates=[]
        self.estimates.append(x0)
         
        # this list is used to store the estimation error covariance matrices Pk
        self.estimationErrorCovarianceMatrices=[]
        self.estimationErrorCovarianceMatrices.append(P0)
        
        # this list is used to store the gain matrices Kk
        self.gainMatrices=[]
         
        # this list is used to store estimation error vectors
        self.errors=[]
    
     
    # this function takes the current measurement and the current measurement matrix C
    # and computes the estimation error covariance matrix, updates the estimate, 
    # computes the gain matrix, and the estimation error
    # it fills the lists self.estimates, self.estimationErrorCovarianceMatrices, self.gainMatrices, and self.errors
    # it also increments the variable currentTimeStep for 1
    
    # measurementValue - measurement obtained at the time instant k
    # C - measurement matrix at the time instant k
    
    def predict(self,measurementValue,C):
        import numpy as np
        
        # compute the L matrix and its inverse, see Eq. 43
        Lmatrix=self.R+np.matmul(C,np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep],C.T))
        LmatrixInv=np.linalg.inv(Lmatrix)
        # compute the gain matrix, see Eq. 42 or Eq. 48
        gainMatrix=np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep],np.matmul(C.T,LmatrixInv))

        # compute the estimation error                    
        error=measurementValue-np.matmul(C,self.estimates[self.currentTimeStep])
        # compute the estimate, see Eq. 49
        estimate=self.estimates[self.currentTimeStep]+np.matmul(gainMatrix,error)
        
        # propagate the estimation error covariance matrix, see Eq. 50            
        ImKc=np.eye(np.size(self.x0),np.size(self.x0))-np.matmul(gainMatrix,C)
        estimationErrorCovarianceMatrix=np.matmul(ImKc,self.estimationErrorCovarianceMatrices[self.currentTimeStep])
        
        # add computed elements to the list
        self.estimates.append(estimate)
        self.estimationErrorCovarianceMatrices.append(estimationErrorCovarianceMatrix)
        self.gainMatrices.append(gainMatrix)
        self.errors.append(error)
        
        # increment the current time step
        self.currentTimeStep=self.currentTimeStep+1
       
        
            
        
        
           
            
    
    
    
    

