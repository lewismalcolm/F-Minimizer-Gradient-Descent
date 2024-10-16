# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 02:35:47 2024

@author: lewis
"""

#Problem 1 part 4 Exact Line Search

import numpy as np

def gd(start, f, gradient, hessian, maxiter, tol=0.00000001):
    step = [start] ## tracking history of x
    step_f = [f(start)]
    x = start
    k = 0  
    for i in range(maxiter):          
            h = hessian(x)
            step_size = (1)/(h)
            diff = step_size*gradient(x)
            if np.all(abs(diff))<tol:               
                    break 
            x = x - diff
            k = k+1
            fc = f(x)
            step_f.append(fc)
            step.append(x) ## tracking
    return step, x

def f(x):
    x1,x2 = x
    return  2*x1**2 + 2*x1*x2 + x2**2 + x1 - x2
def grad_f(x):
    x1, x2 = x
    grad_x1 = 4*x1 + 2*x2 + 1
    grad_x2 = 2*x2 + 2*x1 - 1
    return np.array([grad_x1, grad_x2])
def hess_f(x):
    x1, x2 = x
    hess_x1 = 4
    hess_x2 = 2
    return np.array([hess_x1, hess_x2])
hisory, solution = gd(np.array([1,-1]),f,grad_f,hess_f,23)



#Problem 1 part 5 Fixed Stepsize
import numpy as np

def gd(start, f, gradient, step_size, maxiter, tol=0.01):
    step = [start] ## tracking history of x
    x = start
    k=0
    for i in range(maxiter):
        diff = step_size*gradient(x)
        if np.all(np.abs(diff)<tol):
            break
        
        k=k+1
        x = x - diff
        fc = f(x)
        step.append(x) ## tracking
        
       
    return step, x

def f(x):
    x1,x2 = x
    return  2*x1**2 + 2*x1*x2 + x2**2 + x1 - x2

def grad_f(x):
    x1, x2 = x
    grad_x1 = 4*x1 + 2*x2 + 1
    grad_x2 = 2*x2 + 2*x1 - 1
    return np.array([grad_x1, grad_x2])

history, solution = gd(np.array([4,2]),f,grad_f,0.1,100)


#Problem 1 part 6 Backtracking

import numpy as np
import numpy.linalg as npl
from numpy import linalg as lp

def f(x):
    x1,x2 = x
    return  2*x1**2 + 2*x1*x2 + x2**2 + x1 - x2
    
def df(x):
    x1,x2 = x
    return np.array([4*x1 + 2*x2 + 1, 2*x2 + 2*x1 - 1])
 
def step_size(x):
    alpha = 1
    beta = 0.8
    while f(x - alpha*df(x)) > (f(x) - 0.5*alpha*lp.norm(df(x))**2):
        alpha *= beta
        
    return alpha

def g(lambda_k,x,r):
    return f(x - lambda_k*r)

def steepestdescent(f,df,step_size,x0,tol=1.e-3,maxit=100):
    x = x0
    r = df(x0)
    iters = 0
    while ( np.abs(npl.norm(r))>tol and iters<maxit ):
        lambda_k = step_size(x)
        x = x - lambda_k * r
        r = df(x)
        iters += 1
        
    return x, iters

x0 = np.array([2.0,1.0])
x, iters =steepestdescent(f, df, step_size,x0, tol = 1.e-8, maxit = 100)