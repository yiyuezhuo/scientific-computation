# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:48:36 2016

@author: yiyuezhuo
"""
import numpy as np
#import decomp
from decomp import qr,lr,cholesky
from vows_compiler import matrix_equal

A=np.array([12,-51,4,6,167,-68,-4,24,-41]).reshape(3,3)

Q,R=qr(A)

print('QR')
print(matrix_equal(Q,'Q'))
print(matrix_equal(R,'R'))
print('\n')

A=np.array([[2,3,0],[0,1,0],[4,6,0]])
print('LU')
L,U=lr(A)
print(matrix_equal(L,'L'))
print(matrix_equal(U,'U'))
print('\n')

A=np.array([[4,6,10],[6,58,29],[10,29,38]])
print('cholesky')
T=cholesky(A)
print(matrix_equal(T,'T'))
print('\n')

A=np.diag(np.arange(3))
print('diagonal')
print(matrix_equal(A,'A'))
print('\n')

a=[1,2,3]
b=[4,5,6]
A=np.outer(a,b)
print('outer')
print(matrix_equal(A,'A'))
print('\n')

import statsmodels.api as sm
import scipy.stats as stat

A=np.array([[1,2,3],
            [1,1,0],
            [1,-2,3],
            [1,3,4],
            [1,-10,2],
            [1,4,4],
            [1,10,2],
            [1,3,2],
            [1,4,-1]])
b=np.array([1,-2,3,4,-5,6,7,-8,9])
mod=sm.OLS(b,A)
res=mod.fit()
print(res.summary())

def part(A,sigmahat,i=0):
    index=list(range(A.shape[1]))
    index.remove(i)
    bb=A[:,i]
    AA=A[:,index]
    mod=sm.OLS(bb,AA)
    res=mod.fit()
    SST=res.ess+res.ssr
    R2=res.rsquared
    #sigmahat=np.std(res.resid)
    return sigmahat/np.sqrt(SST*(1-R2))
    
def F_pvalue(x,n1,n2):
    return stat.beta(n1/2,n2/2).cdf(x/(n2/n1+x))
