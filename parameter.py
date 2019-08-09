# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:51:38 2019

@author: ZhangZhaomin
"""
import numpy as np


epoch=200
RF=6
feature=4
Imin = 0
Imax = 1
mu= (Imin + (2*np.arange(1, RF+1)-3) / 2*(Imax-Imin)/(RF-2))
sigma= (1/0.7*(Imax-Imin)/(RF-2))
T_pre=300
TPost=400
class_num=3
tau=3
stdp=1.6
T_train=np.array(range(T_pre+1))
TID=200
T_post= np.matmul(np.ones((feature*RF, 1)), np.arange(400+1).reshape(1, int(400+1)))
T_divide=5
lamada=0.5