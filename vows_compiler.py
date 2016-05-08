# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:44:02 2016

@author: yiyuezhuo
"""

def matrix_equal(mat,mat_name='mat',tol_name='tol',tol_value=0.0001):
    sl=['var {tol_name}={tol_value}'.format(tol_name=tol_name,tol_value=tol_value)]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sl.append('assert.epsilon({tol_name},{mat_name}[{i}][{j}],{value});'.format(tol_name=tol_name,mat_name=mat_name,value=mat[i][j],i=i,j=j))
    return '\n'.join(sl)
    
