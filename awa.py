# -*- coding: utf-8 -*-

import pickle
from SAE import SAE
from NormalizeFea import NormalizeFea
import numpy as np
from math import *
from label_matrix import label_matrix

from scipy import spatial
import scipy.io as sio
# a=scio.loadmat(os.getcwd()+'/'+"awa_demo_data.mat")
X_tr = sio.loadmat('./DatasetB_20180919/X_tr.mat')['X_tr']


X_te = sio.loadmat('./DatasetB_20180919/X_te.mat')['X_te']

X_tr=NormalizeFea(X_tr.T, 2).T

S_tr = sio.loadmat('./DatasetB_20180919/S_tr.mat')['S_tr']

S_te_gt = sio.loadmat('./DatasetB_20180919/word_embedding2.mat')['word_embedding2']

Y_te = sio.loadmat('./DatasetB_20180919/Y_te.mat')['Y_te']


#fea=fea.dot(W)
#fea_tes=fea_tes.dot(W)
lamb = 500000;

W=SAE(X_tr.T, S_tr.T, lamb).T


##计算余弦距离并标准化 
S_te_est = X_te .dot(NormalizeFea(W, 2))
dist     =  1 - spatial.distance.cdist(S_te_est,S_te_gt, 'cosine')
#dist     = NormalizeFea(dist,0)
#[F --> S], projecting data from feature space to semantic space 
HITK=1
Y_hit5 = np. zeros((dist.shape[0], HITK))
for i in range(dist.shape[0]):
    I = np.argsort(dist[i])[::-1]
    Y_hit5[i] = I[166]


n=0
for i in range(dist.shape[0]):
    if Y_te[i] in Y_hit5[i]:
        n=n+1

zsl_accuracy = n/dist.shape[0]
print(n)
#[S --> F], projecting from semantic to visual space 
#dist    =  1 - zscore(pdist2(X_te, (S_te_pro * W'), 'cosine')) ;
# S_te_pro=NormalizeFea(S_te_pro.T,2).T
# dist     =  1 - spatial.distance.cdist(X_te,S_te_pro.dot(W.T),'cosine')
#
# HITK=1
# Y_hit5 =np. zeros((dist.shape[0],HITK))
# for i in range(dist.shape[0]):
#     I=np.argsort(dist[i])[::-1]
#     Y_hit5[i,:]=te_cl_id[I[0:HITK]]
#
#
# n=0
# for i in range(dist.shape[0]):
#     if Y_te[i] in Y_hit5[i,:]:
#         n=n+1
#
# zsl_accuracy = n/dist.shape[0]
# print(zsl_accuracy)



     

