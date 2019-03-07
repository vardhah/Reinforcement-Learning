# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:31:23 2019

@author: HPP
"""

import numpy as np
import matplotlib as plt
import sklearn as sk

r=-1
gamma=1
v=np.zeros((4,4))
delta=0
print(v.shape)
v_old=np.zeros((4,4))
for i in range(10):
    v_old[:]=v
    print('v_old:\n', v_old)
    v[1]=(1/4)*(r+gamma*v_old[0])+(1/4)*(r+gamma*v_old[2])+(1/4)*(r+gamma*v_old[5])+(1/4)*(r+gamma*v_old[1])
    v[2]=(1/4)*(r+gamma*v_old[1])+(1/4)*(r+gamma*v_old[3])+(1/4)*(r+gamma*v_old[6])+(1/4)*(r+gamma*v_old[2])
    v[3]=(1/4)*(r+gamma*v_old[2])+(1/4)*(r+gamma*v_old[3])+(1/4)*(r+gamma*v_old[7])+(1/4)*(r+gamma*v_old[3])
    v[4]=(1/4)*(r+gamma*v_old[4])+(1/4)*(r+gamma*v_old[5])+(1/4)*(r+gamma*v_old[0])+(1/4)*(r+gamma*v_old[8])
    v[5]=(1/4)*(r+gamma*v_old[4])+(1/4)*(r+gamma*v_old[6])+(1/4)*(r+gamma*v_old[1])+(1/4)*(r+gamma*v_old[9])
    v[6]=(1/4)*(r+gamma*v_old[5])+(1/4)*(r+gamma*v_old[7])+(1/4)*(r+gamma*v_old[2])+(1/4)*(r+gamma*v_old[10])
    v[7]=(1/4)*(r+gamma*v_old[6])+(1/4)*(r+gamma*v_old[7])+(1/4)*(r+gamma*v_old[3])+(1/4)*(r+gamma*v_old[11])
    v[8]=(1/4)*(r+gamma*v_old[8])+(1/4)*(r+gamma*v_old[9])+(1/4)*(r+gamma*v_old[4])+(1/4)*(r+gamma*v_old[12])
    v[9]=(1/4)*(r+gamma*v_old[8])+(1/4)*(r+gamma*v_old[10])+(1/4)*(r+gamma*v_old[5])+(1/4)*(r+gamma*v_old[13])
    v[10]=(1/4)*(r+gamma*v_old[9])+(1/4)*(r+gamma*v_old[11])+(1/4)*(r+gamma*v_old[6])+(1/4)*(r+gamma*v_old[14])
    v[11]=(1/4)*(r+gamma*v_old[10])+(1/4)*(r+gamma*v_old[11])+(1/4)*(r+gamma*v_old[7])+(1/4)*(r+gamma*v_old[15])
    v[12]=(1/4)*(r+gamma*v_old[12])+(1/4)*(r+gamma*v_old[13])+(1/4)*(r+gamma*v_old[8])+(1/4)*(r+gamma*v_old[12])
    v[13]=(1/4)*(r+gamma*v_old[12])+(1/4)*(r+gamma*v_old[14])+(1/4)*(r+gamma*v_old[9])+(1/4)*(r+gamma*v_old[13])
    v[14]=(1/4)*(r+gamma*v_old[13])+(1/4)*(r+gamma*v_old[15])+(1/4)*(r+gamma*v_old[10])+(1/4)*(r+gamma*v_old[14])
    print('v is:',v)