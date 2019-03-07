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
print('=> Reward matrix is:\n',r)

#Map coordinate grid to location grid(generated mapper)[can be used for any 2D game]
statemap=np.ones((4,4))
statemap=statemap.astype(int)
mapper=np.zeros((4,4))
mapper=mapper.astype(int)
k=0
for i in range(statemap.shape[0]):
  for j in range(statemap.shape[1]):
   mapper[i][j]=k+j
  k=k+j+1
print('\n=> Mapper matrix is:\n',mapper)



s=12 # represent state of agent
a=0 # a represent action ( 0 for => , 1 for up, 2 for <= , 3 for down)


def q_learning(epsilon=0.25,episodes=1000,simulation=5):
 global a,s
 alpha=0.1
 gamma= 0.9
 
 print('Epsilon is:',epsilon,'\nNumber of Episodes are:',episodes)
 
 reinforcement={}      #Dictionary to store reinforcement array for each simulation run
 iteration=[]          #count number of episodes
 states=[]             #used for debugging purpose ( to know the states navigated)
 rein=[]               # collect reinforecemnt for a simulation in array
 reinforce=0           # collect Reinforcement for a run 
 
 for j in range(simulation):
  q=np.zeros((16,4))   #Initialisation of Q matrix
  rein=[]              #clear rein array after each simulation
  iteration=[] 
  countexplore,countexploit = 0,0        
  for i in range(episodes):
   s=12                #Intialise state after every run 
   states=[]           #clear state array
   reinforce=0         #clear reinforcement after every run
   
   while (s!=0 and s!=15):      #until reached state 0 ( one is labled 0 & other is 15 )
     states.append(s)
     z=np.random.rand(1)        #generate a random samples from a uniform distribution over [0, 1).
     if(z<epsilon):             #test, if generated number is less than exploration rate
      countexplore+=1
      a=np.random.randint(4)    #generate random integer from a uniform distribution over [0, 4).random move
     else:
      countexploit+=1
      a=np.argmax(q[s])         #take action based on learning(maximise)   
     
     s_p= updatestate(s,a)      #calculate the next state and its location
     reinforce+=r[s,a]          #add the reinforcement received
     q[s,a]=q[s,a]+alpha*(r[s,a]+gamma*np.amax(q[s_p])-q[s,a])       # Q value updation
     s=s_p                                                           # Update state
   
   rein.append(reinforce)                    # once run is complete, add received reinforcemnt to rein array
   iteration.append(i)
  
  reinforcement[j]=np.array(rein)           #once episode is done add it in dictionary 
  print('-========Simulation',j+1,'====================================-')
  print('explored ',countexplore,'times')
  print('exploited ',countexploit,'times')
 
 reinforcementC=reinforcement[0]
 for count in range(1,len(reinforcement)):
     reinforcementC= np.vstack((reinforcementC,reinforcement[count]))
 #print(reinforcementC)
 reinforcementC=np.average(reinforcementC,axis=0)
 
 plt.plot(iteration,reinforcementC,color='green',linewidth=1)
 

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
    
    