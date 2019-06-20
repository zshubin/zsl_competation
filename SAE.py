# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:55:03 2018

@author: LiZhiHuan
"""
from scipy import linalg
import numpy as np
def SAE(X,S,lamb):

    A=np.nan_to_num(S.dot(S.T))
    B=np.nan_to_num(lamb*(X.dot(X.T)))
    C=np.nan_to_num((1+lamb)*(S.dot(X.T)))
    W=linalg.solve_sylvester(A,B,C)
    return W
    

    