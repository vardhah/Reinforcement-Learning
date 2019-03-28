import numpy as np 
import matplotlib.pyplot as plt

"""Reward used for example 2 assignment 
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
"""

#Map coordinate grid to location grid(generated mapper)[can be used for any 2D game]
def_size= 4    # select size of grid
num_reward=2   # select number of gold we want on grid 

statemap=np.ones((def_size,def_size))
statemap=statemap.astype(int)
mapper=np.zeros((def_size,def_size))
mapper=mapper.astype(int)
k=0
for i in range(statemap.shape[0]):
  for j in range(statemap.shape[1]):
   mapper[i][j]=k+j
  k=k+j+1
print('\n=> Mapper matrix is:\n',mapper)
total= statemap.shape[0]*statemap.shape[1]
start1 = 0 
start2= (total-statemap.shape[0])
end= (total-1)
print('start1 is:',start1)
print('start2 is:',start2)
print('End is:',end)
#gold_p=(np.random.randint(1,(start2-1),size=(1,num_reward)))
gold_p=np.random.choice(range(1,start2-1), num_reward, replace=False)
print('positon of gold is:',gold_p)
r=np.ones((total,4))*-1 

#define reward matrix , reward is 1 for gold, -1 for state transition , 0 for end point
def reward(gold_p):  
 global end,total,r    
 r=np.ones((total,4))*-1
 #modifying the reward function for end point
 endby=findnearby(end)
 for j in range(len(endby)):
  if (endby[j]==end):
    continue
  else:
    index=(j+2)%4
    r[endby[j]][index]=1
 #print('End by is:',endby)
 
 for i in range(len(gold_p)):
    #print('location we are working on is:',gold_p[i])
    nearby=findnearby(gold_p[i])
    #print(nearby)
    for j in range(len(nearby)):
        if (nearby[j]==gold_p[i]):
            continue
        else:
            index=(j+2)%4
            r[nearby[j]][index]=1
        
 #print('updated reward is:=>\n',r)


a1=0          # a represent action ( 0 for => , 1 for up, 2 for <= , 3 for down)
a2=0          # a1 for action of 1st agent, a2 is action for 2nd agent. 
s1= start1    # Represent current state of agent1
s2=start2     # Represent present state of agent2


def hyst_learning(epsilon=0.25,alpha=0.1,beta=0.01,episodes=20, simulation=4):
 global a1,a2,s1,s2,start1,start2,gold_p,r,total
 delta1 =0 
 delta2=0
 gamma=0.9
 
 print('Epsilon is:',epsilon,'\nNumber of Episodes are:',episodes)
 #print('Gamma is:',gamma)
 print('Alpha is:',alpha)
 print('Alpha is:',alpha)
 reinforcement={}      #Dictionary to store reinforcement array for each simulation run
 iteration=[]          #count number of episodes
 states1,states2=[] ,[]            #used for debugging purpose ( to know the states navigated)
 rein=[]               # collect reinforecemnt for a simulation in array
 #reinforce=0           # collect Reinforcement for a run 
 
 for j in range(simulation):
  q1=np.zeros((total,4))   #Initialisation of Q1 matrix
  q2=np.zeros((total,4))   #Initialisation of Q1 matrix
  rein=[]                 #clear rein array after each simulation
  iteration=[] 
  countexplore,countexploit = 0,0        
  for i in range(episodes):
   print('======>>>>starting episode',i)
   s1=start1
   s2=start2                         #Intialise state after every run 
   gold_l=np.array(gold_p)           # initialise gold list
   states1,states2=[],[]             #clear state array
   reinforce1,reinforce2=0,0         #clear reinforcement after every run

   #starting simulation run  ------------
   while (s1!=end or s2!=end or len(gold_l)!=0):    # until agent1 & agent2 reaches end and no gold left
     states1.append(s1)
     states2.append(s2)
    # deploying epsilon greedy 
     z=np.random.rand(1)             #generate a random samples from a uniform distribution over [0, 1).
     if(z<epsilon):                  #test, if generated number is less than exploration rate
      countexplore+=1
      a1=np.random.randint(4)        #generate random integer from a uniform distribution over [0, 4).random move
      a2=np.random.randint(4) 
     else:
      countexploit+=1
      a1=np.argmax(q1[s1])         #take action based on learning(maximise)   
      a2=np.argmax(q2[s2])         #take action based on learning(maximise)  
     
     #print('a1 is:',a1)
     #print('a2 is:',a2)
     
     # Crash situation handler 
     if (s1==s2 and s1!=end and s2!=end):
         print('crash')
         break
      
     #print('gold_l is',gold_l)
     #check if reached gold position => update gold list & call reward updation function        
     if s1 in gold_l:
         print('Found gold by 1:',s1)
         position=np.where(gold_l==s1)
         print(position)
         gold_l=np.delete(gold_l,position)
         print('gold_l is:',gold_l)
         reward(gold_l)
     if s2 in gold_l:
         print('found gold by 2:',s2)
         position=np.where(gold_l==s2)
         print(position)
         gold_l=np.delete(gold_l,position)
         print('gold_l is:',gold_l)
         reward(gold_l)
       
     s_p1= updatestate(s1,a1)      #calculate the next state and its location
     s_p2= updatestate(s2,a2)     
     reinforce1+=r[s1,a1]             #add the reinforcement received
     reinforce2+=r[s2,a2]
     delta1= r[s1,a1]+gamma*np.amax(q1[s_p1])-q1[s1,a1] 
     delta2= r[s2,a1]+gamma*np.amax(q2[s_p2])-q2[s2,a2] 
     if delta1>=0:
      q1[s1,a1]=q1[s1,a1]+alpha*delta1      # Q value updation
     elif delta1<0: 
      q1[s1,a1]=q1[s1,a1]+beta*delta1       # Q value updation   
     if delta2>=0:
      q2[s2,a2]=q2[s2,a2]+alpha*delta2      # Q value updation
     elif delta2<0: 
      q2[s2,a2]=q2[s2,a2]+beta*delta2       # Q value updation    
     s1=s_p1
     s2=s_p2                              # Update state
   
   rein.append(reinforce1+reinforce2)                    # once run is complete, add received reinforcemnt to rein array
   iteration.append(i)
   print('states1:',states1)
   print('states2:',states2)
  reinforcement[j]=np.array(rein)           #once episode is done add it in dictionary 
  print('-========Simulation',j+1,'====================================-')
  print('explored ',countexplore,'times')
  print('exploited ',countexploit,'times')
  print('q1 is',q1)
  print('q2 is',q2)
 reinforcementC=reinforcement[0]
 for count in range(1,len(reinforcement)):
     reinforcementC= np.vstack((reinforcementC,reinforcement[count]))
 #print(reinforcementC)
 reinforcementC=np.average(reinforcementC,axis=0)
 
 plt.plot(iteration,reinforcementC,color='green',linewidth=1)
 

def hyst_c_f1(epsilon=0.1,alpha=0.1,beta=0.01,episodes=1000):
  r=np.ones((3,3))
  r[0,0]=11
  r[0,1]=-30
  r[0,2]=0
  r[1,0]=-30
  r[1,1]=7
  r[1,2]=6
  r[2,0]=0
  r[2,1]=0
  r[2,2]=5
  print('=> Reward matrix is:\n',r)  
  q1=np.zeros((1,3))   #Initialisation of Q1 matrix
  q2=np.zeros((1,3))
  print('q1 is:',q1)
  print('q2 is:',q2)
  
  delta1 =0 
  delta2=0
  gamma=0.9
  
  for i in range(episodes):
      

     #print('======>>>>starting episode',i)
# deploying epsilon greedy 
     z=np.random.rand(1)             #generate a random samples from a uniform distribution over [0, 1).
     if(z<epsilon):                  #test, if generated number is less than exploration rate
      #countexplore+=1
      a1=np.random.randint(3)        #generate random integer from a uniform distribution over [0, 4).random move
      a2=np.random.randint(3) 
     else:
      #countexploit+=1
      a1=np.argmax(q1)         #take action based on learning(maximise)   
      a2=np.argmax(q2)         #take action based on learning(maximise)  
   
  # NOt generating random number  
   #a1=np.random.randint(3)        #generate random integer from a uniform distribution over [0, 3).random move
   #a2=np.random.randint(3)
   
     #print('a1 is:',a1,'a2 is:',a2)
     delta1= r[a1,a2]+gamma*np.amax(q1)-q1[0,a1] 
     delta2= r[a1,a2]+gamma*np.amax(q2)-q2[0,a2]
     if delta1>=0:
       q1[0,a1]=q1[0,a1]+alpha*delta1      # Q value updation
     elif delta1<0: 
       q1[0,a1]=q1[0,a1]+beta*delta1       # Q value updation   
     if delta2>=0:
       q2[0,a2]=q2[0,a2]+alpha*delta2      # Q value updation
     elif delta2<0: 
       q2[0,a2]=q2[0,a2]+beta*delta2       # Q value updation 
  print('final Q1 value is:',q1)
  print('final Q2 value is:',q2)





def hyst_p_f1(epsilon=0.25,alpha=0.1,beta=0.01,episodes=1000):
  r=np.ones((3,3))
  k=-1
  r[0,0]=10
  r[0,1]=0
  r[0,2]=k
  r[1,0]=0
  r[1,1]=2
  r[1,2]=0
  r[2,0]=k
  r[2,1]=0
  r[2,2]=10
  print('=> Reward matrix is:\n',r)  
  q1=np.zeros((1,3))   #Initialisation of Q1 matrix
  q2=np.zeros((1,3))
  print('q1 is:',q1)
  print('q2 is:',q2)
  
  delta1 =0 
  delta2=0
  gamma=0.9
  
  for i in range(episodes):
      

     #print('======>>>>starting episode',i)
# deploying epsilon greedy 
     z=np.random.rand(1)             #generate a random samples from a uniform distribution over [0, 1).
     if(z<epsilon):                  #test, if generated number is less than exploration rate
      #countexplore+=1
      a1=np.random.randint(3)        #generate random integer from a uniform distribution over [0, 4).random move
      a2=np.random.randint(3) 
     else:
      #countexploit+=1
      a1=np.argmax(q1)         #take action based on learning(maximise)   
      a2=np.argmax(q2)         #take action based on learning(maximise)  
   
  # NOt generating random number  
   #a1=np.random.randint(3)        #generate random integer from a uniform distribution over [0, 3).random move
   #a2=np.random.randint(3)
   
     #print('a1 is:',a1,'a2 is:',a2)
     delta1= r[a1,a2]+gamma*np.amax(q1)-q1[0,a1] 
     delta2= r[a1,a2]+gamma*np.amax(q2)-q2[0,a2]
     if delta1>=0:
       q1[0,a1]=q1[0,a1]+alpha*delta1      # Q value updation
     elif delta1<0: 
       q1[0,a1]=q1[0,a1]+beta*delta1       # Q value updation   
     if delta2>=0:
       q2[0,a2]=q2[0,a2]+alpha*delta2      # Q value updation
     elif delta2<0: 
       q2[0,a2]=q2[0,a2]+beta*delta2       # Q value updation 
  print('final Q1 value is:',q1)
  print('final Q2 value is:',q2)
  
  
  
  
  
  
  
  
  

def q_learning(epsilon=0.25,episodes=500,simulation=5):
 global a,s
 alpha=0.1
 gamma= 0.9
 
 print('Epsilon is:',epsilon,'\nNumber of Episodes are:',episodes)
 #print('Gamma is:',gamma)
 print('Alpha is:',alpha)
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
  
  #customise the experiment 1,2 & 3  (for exp 1,comment all 3 lines)(exp3=>comment "alpha=1/i" ) 
   if i!=0:
    epsilon = 1/i
    alpha=1/i
   
   #starting simulation run  ------------
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
 
#below function can be called to update state by given current state and action 
# @parameter : ( state, action)
def updatestate(s,a):    
 global end
 if (s==end):
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

#Below function convert loaction(labled in number) to coordinate system 
# @param (location label)
def fetchcoord(loc):
    z=np.where(mapper==loc)
    x=np.concatenate((z[0], z[1]), axis=0)
    return x

#Below function convert coordinate system in to location(labled in number). 
def fetchloc(coord):
    return mapper[coord[0]][coord[1]]
 
def findnearby(loc):
     #print('current location is:',loc)
     global def_size
     max_c=def_size-1
     #print(max_c)
     cc=fetchcoord(loc)
     neighbour=[]
     if(cc[1]!=max_c):
         uc=cc[1]+1  
         neighbour.append(fetchloc([cc[0],uc]))
     if(cc[1]==max_c):
         neighbour.append(fetchloc(cc))
     
     if(cc[0]!=0):
         uc=cc[0]-1   
         neighbour.append(fetchloc([uc,cc[1]])) 
     if(cc[0]==0):
         neighbour.append(fetchloc(cc))
         
     if(cc[1]!=0):
         uc=cc[1]-1   
         neighbour.append(fetchloc([cc[0],uc])) 
     if(cc[1]==0):
         neighbour.append(fetchloc(cc))
         
     if(cc[0]!=max_c):
         uc=cc[0]+1       
         neighbour.append(fetchloc([uc,cc[1]])) 
     if(cc[0]==max_c):
         neighbour.append(fetchloc(cc))    
     return neighbour