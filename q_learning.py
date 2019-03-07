# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:00:17 2019

@author: HPP
"""

import numpy as np 
import matplotlib.pyplot as plt


r=np.ones((16,4))
r=r*-1
r[1,2]=1
r[4,1]=1
r[11,3]=1
r[14,0]=1
r[0,1]=1
r[0,2]=1
r[15,0]=1
r[15,3]=1

print('Reward matrix is:\n',r)

#Map coordinate grid to location grid
statemap=np.ones((4,4))
statemap=statemap.astype(int)
mapper=np.zeros((4,4))
mapper=mapper.astype(int)
print('Initial grid map is:\n',statemap)
k=0
for i in range(statemap.shape[0]):
  for j in range(statemap.shape[1]):
   mapper[i][j]=k+j
  k=k+j+1
 
print('Mapper matrix is:\n',mapper)



q=np.zeros((16,4))
print('Q matrix is:\n',q)
s=12
a=0 # a 


def epsilon_greedy(epsilon=0.25,numberoftrails=1000,episode=5):
 global a
 global s
 alpha=0.1
 gamma= 0.9
 countexplore = 0
 countexploit=0
 print('Epsilon is:',epsilon,'\nNumer of trails are:',numberoftrails)
 reinforcement={}
 iteration=[]
 states=[]
 reinforce=0
 for j in range(episode):
  q=np.zeros((16,4))   
  for i in range(numberoftrails):
   s=12
   #print('States Navigated is:',states)
   #print('Reinforcement:',reinforce)
   states=[]
   reinforce=0
   while (s!=0 and s!=15):
     states.append(s)
     #print('current location is :',s) 
     z=np.random.rand(1)
     if(z<epsilon):
      countexplore+=1
      a=np.random.randint(4)
      #print('------------------------------------------')
      #print('Random action would be:',a)
      #print('Reward would be:',r[s,a])
     else:
      countexploit+=1
      a=np.argmax(q[s])
      #print('------------------------------------------')
      #print('Max rewarded Action is:',a) 
      #print('Reward is:',np.amax(r[s]))  
     s_p= updatestate(s,a)
     reinforce+=r[s,a]
     #print(np.amax(q[s_p]))
     q[s,a]=q[s,a]+alpha*(r[s,a]+gamma*np.amax(q[s_p])-q[s,a])
     s=s_p
   reinforcement
   reinforcement[j].append(reinforce)
   iteration.append(i)
  
  #print('-============================================-')
  #print('explored ',countexplore,'times')
  #print('exploited ',countexploit,'times')
  #print('Final Q is:',q)   
 #plt.plot(iteration,reinforcement,color='green',linewidth=1)
 #print('Q matrix is:\n',q)
 
def updatestate(s,a):
 
 if (s==0 or s==15):
     #print('Navigation over')
     return s
 else: 
     cc=fetchcoord(s)
     if (a==0):
         #print('action is',a)
         if(cc[1]!=3):
           cc[1]+=1                
     if (a==1):
         if(cc[0]!=0):
           cc[0]-=1   
         #print('action is',a)
     if (a==2):
         if(cc[1]!=0):
           cc[1]-=1   
         #print('action is',a)
     if (a==3):
         if(cc[0]!=3):
           cc[0]+=1   
         #print('action is',a)
     #print('current coordinate is:',cc)
     zz= fetchloc(cc)
     #print('Updated location is',zz)
     return zz    
 
def fetchcoord(loc):
    z=np.where(mapper==loc)
    x=np.concatenate((z[0], z[1]), axis=0)
    return x

def fetchloc(coord):
    return mapper[coord[0]][coord[1]]

def greedysearch(m,st,f):
    #m => final output matrix(q) ; i=> initial stage; f=> final stage
    state=st
    #print(f[0])
    #print(f[1])
    while( state!=f[0] and state !=f[1]):
     print(q[state]) 
     print(np.argmax(q[state]))
     direction=np.argmax(q[state])
     state=updatestate(state,direction)
     print('current state is:',state)
    
    