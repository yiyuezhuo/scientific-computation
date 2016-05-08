# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:39:59 2016

@author: yiyuezhuo
"""

import numpy as np
import scipy
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Latex,Math
#%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
sh = InteractiveShell.instance()

def number_to_str(n,cut=5):
    ns=str(n)
    format_='{0:.'+str(cut)+'f}'
    if 'e' in ns or ('.' in ns and len(ns)>cut+1):
        return format_.format(n)
    else:
        return str(n)

def matrix_to_latex(mat,style='bmatrix'):
    if type(mat)==np.matrixlib.defmatrix.matrix:
        mat=mat.A
    head=r'\begin{'+style+'}'
    tail=r'\end{'+style+'}'
    if len(mat.shape)==1:
        body=r'\\'.join([str(el) for el in mat])
        return head+body+tail
    elif len(mat.shape)==2:
        lines=[]
        for row in mat:
            lines.append('&'.join([number_to_str(el)  for el in row])+r'\\')
        s=head+' '.join(lines)+tail
        return s
    return None

sh.display_formatter.formatters['text/latex'].type_printers[np.ndarray]=matrix_to_latex

def show_decomposition(*args):
    latex=''
    for arg in args:
        if type(arg)==str:
            latex+=arg
        else:
            latex+=matrix_to_latex(arg)
    latex='$'+latex+'$'
    display(Math(latex))
    

def subsup(A,b):
    size=A.shape[1]
    x=np.zeros(size)
    for i in range(size-1,-1,-1):
        x[i]=(b[i]-sum([x[j]*A[i][j] for j in range(i+1,size)]))/A[i][i]   
    return x

def subsdown(A,b):
    size=A.shape[1]
    x=np.zeros(size)
    for i in range(size):
        x[i]=(b[i]-sum([A[i][j]*x[j] for j in range(i)]))/A[i][i]
    return x
    
def lr(A):
    size=A.shape[0]
    L=np.diag(np.ones(size))
    R=np.zeros(A.shape)
    for t in range(size):
        R[0][t]=A[0][t]
    for l in range(1,size):
        for i in range(l):
            L[l][i]=(A[l][i]-sum([L[l][jj]*R[jj][i] for jj in range(i) ]))/R[i][i]
        for j in range(l,size):
            R[l][j]=A[l][j]-sum([L[l][jj]*R[jj][j] for jj in range(l)])
    return L,R

def solve_lr(A,b):
    L,R=lr(A)
    y=subsdown(L,b)
    x=subsup(R,y)
    return x
    
def ldrstar(A):
    size=A.shape[0]
    L,R=lr(A)
    #Rstar=np.zeros([size,size])
    Rstar=np.identity(size)
    D=np.diag(R.diagonal())
    for i in range(size-1):
        for j in range(i+1,size):
            Rstar[i][j]=R[i][j]/D[i][i]
    return L,D,Rstar

def lstarrstar(A):
    L,D,Rstar=ldrstar(A)
    Lstar=np.zeros(L.shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            Lstar[i][j]=L[i][j]*D[j][j]
    return Lstar,Rstar

def cholesky(A):
    size=A.shape[0]
    T=np.zeros(A.shape)
    for i in range(size):
        T[i][i]=np.sqrt(A[i][i]-sum([T[i][t]**2 for t in range(i)]))
        for j in range(i+1,size):
            T[j][i]=(A[i][j]-sum([T[i][t]*T[j][t] for t in range(i)]))/T[i][i]
    return T
    
def get_Q1(x):
    size=x.shape[0]
    norm_x=np.sqrt(np.dot(x,x))
    e1=np.zeros(size)
    e1[0]=1
    u=x-norm_x*e1
    norm_u=np.sqrt(np.dot(u,u))
    v=u/norm_u
    Q=np.identity(size)-2*np.outer(v,np.transpose(v))
    return Q

from functools import reduce

def qr(A):
    size=A.shape[1]
    QList=[]
    for i in range(size):
        x=A[i:,i]
        Q=get_Q1(x)
        Qn=np.identity(A.shape[0])
        #print(Qn.shape,Q.shape)
        Qn[i:,i:]=Q
        A=np.dot(Qn,A)
        QList.append(Qn)
    Q=reduce(lambda x,y:x.dot(y),QList)
    R=A
    return Q,R

def R_I(A):
    #上三角矩阵R -> R^I
    A=A.copy()
    size=A.shape[0]
    I=np.identity(size)
    for i in range(size-1,-1,-1):
        I[i,:]=I[i,:]/A[i][i]
        A[i,:]=A[i,:]/A[i][i]
        for j in range(i):
            c=-A[j][i]
            A[j,:]=A[j,:]+c*A[i,:]
            I[j,:]=I[j,:]+c*I[i,:]
    return I
    
def qr_solve(A,b):
    Q,R=qr(A)
    attrs=A.shape[1]
    Q1=Q[:,:attrs]
    R1=R[:attrs,:]
    RI=R_I(R1)
    return RI.dot(np.transpose(Q1)).dot(b)

A=np.array([12,-51,4,6,167,-68,-4,24,-41]).reshape(3,3)

Q,R=qr(A)



#show_decomposition(Q.dot(R),'=',Q,R)