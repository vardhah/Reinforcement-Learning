# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:31:23 2019

@author: Harsh Vardhan 
"""

import numpy as np
import matplotlib as plt
import sklearn as sk

v_old=np.zeros((4,4))

def value_iteration(iterator=10):
 r=-1
 gamma=1
 global v_old
 v=np.zeros((4,4))
 #delta=0
 print(v.shape)
 v_old=np.zeros((4,4))
 for i in range(iterator):
    v_old[:]=v
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
          if (i==0 and j==0):
              continue
              print(v[i][j])
          if(i==3 and j==3):
              continue
              print(v[i][j])
          else:
            v[i][j]= 1/4*(r+gamma*rightV(i,j))+1/4*(r+gamma*leftV(i,j))+1/4*(r+gamma*upV(i,j))+1/4*(r+gamma*downV(i,j))
    print('v is:\n',v)

def rightV(x,y):
    if(y!=3):
           y+=1
    return v_old[x][y]

def leftV(x,y):
    if(y!=0):
           y-=1 
    return v_old[x][y]
    
def upV(x,y):
    if(x!=0):
           x-=1 
    return v_old[x][y]
    
def downV(x,y):
    if(x!=3):
           x+=1   
    return v_old[x][y]
   
    
   
    