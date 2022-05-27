##################################################
# Tuning
# Julie Butler Hartley
# Version 0.5.0
# Date Created: May 27, 2022
# Last Modified: May 27, 2022
# 
# A set of functions for tuning ridge regression and kernel ridge regression
# alorithms, both with and without the sequential regression extrapolation
# formatting
##################################################


##############################
##         IMPORTS         ##
##############################
import numpy as np
from Extrapolate import *
from Regression import *
from Support import *


##############################
##  TUNE RIDGE REGRESSION   ##
##############################
def tune_ridge_regression(x_train, y_train, x_test, y_test, alpha_range = [0,1e-6,1e-4,1e-2,1,10,100], verbose=True):
    best_error = 100
    best_alpha = 55
    for alpha in alpha_range:
        R = RR(alpha=alpha)
        R.fit(x_train,y_train)
        y_pred = R.predict(x_test)
        err = rmse(np.array(y_pred), np.array(y_test))
        if err < best_err:
            best_err = err
            best_alpha = alpha
    if verbose:
        print("The best RMSE error is" best_err, "with an alpha of", best_alpha)
    return best_err, best_alpha


########################################
##      TUNE RIDGE REGRESSION SRE     ##
########################################
def tune_ridge_regression_sre(X,Y, training_dim, len_extrapolate, 
                              alpha_range = [0,1e-6,1e-4,1e-2,1,10,100], 
                              seq_range=[1,2,3,4,5], verbose=True):
    print("TO BE IMPLEMENTED")
    
    
##################################################
##    TIME POLYNOMIAL KERNEL RIDGE REGRESSION   ##
##################################################
def tune_polynomial_kernel_ridge_regresssion(x_train, y_train, x_test, y_test,gamma_range=np.arange(-5.0,5.5,0.5), 
                                             c0_range=np.arange(-5.0,5.5,0.5),p_range=[1,2,3,4],
                                             alpha_range=[0,1e-6,1e-4,1e-2,1,10,100], verbose=True):
    best_err = 100
    best_alpha = 55
    best_params = []
    for gamma in gamma_range:
        for c0 in c0_range:
            for p in p_range:
                for alpha in alpha_range:
                    R = KRR(params=[gamma, c0, p], kernel_func="p", alpha=alpha)
                    R.fit(x_train, y_train)
                    y_pred = R.predict(x_test)
                            err = rmse(np.array(y_pred), np.array(y_test))
                            if err < best_err:
                                best_err = err
                                best_alpha = alpha
                                best_params=[gamma,c0,p]
    if verbose:
        print("The best RMSE error is" best_err, "with an alpha of", best_alpha,"and params of",best_params)
    return best_err, best_alpha, best_params


##################################################
## TUNE POLYNOMIAL KERNEL RIDGE REGRESSION SRE  ##
##################################################
def tune_polynomial_kernel_ridge_regresssion_sre(X,Y,training_dim,len_extrapolate,gamma_range=np.arange(-5.0,5.5,0.5), 
                                             c0_range=np.arange(-5.0,5.5,0.5),p_range=[1,2,3,4],
                                             alpha_range=[0,1e-6,1e-4,1e-2,1,10,100], seq_range=[1,2,3,4,5],verbose=True):
    print("To be implemented")    
                    
