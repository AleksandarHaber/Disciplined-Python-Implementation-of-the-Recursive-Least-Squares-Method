# -*- coding: utf-8 -*-
"""
This is the main file that demonstrates how to use the recursive least squares method 
implemented in RecursiveLeastSquares.py
The background information and explanation of this code is given here 

https://aleksandarhaber.com/introduction-to-kalman-filter-disciplined-python-implementation-of-recursive-least-squares-method/
"""

import numpy as np
import matplotlib.pyplot as plt
from RecursiveLeastSquares import RecursiveLeastSquares
# define the true parameters that we want to estimate

# true value of the parameters that will be estimated 
initialPosition=10
acceleration=5
initialVelocity=-2

# noise standard deviation
noiseStd=1;

# simulation time
simulationTime=np.linspace(0,15,2000)
# vector used to store the somulated position
position=np.zeros(np.size(simulationTime))

# simulate the system behavior
for i in np.arange(np.size(simulationTime)):
    position[i]=initialPosition+initialVelocity*simulationTime[i]+(acceleration*simulationTime[i]**2)/2

# add the measurement noise 
positionNoisy=position+noiseStd*np.random.randn(np.size(simulationTime))

# verify the position vector by plotting the results
plotStep=300
plt.plot(simulationTime[0:plotStep],position[0:plotStep],linewidth=4, label='Ideal position')
plt.plot(simulationTime[0:plotStep],positionNoisy[0:plotStep],'r', label='Observed position')
plt.xlabel('time')
plt.ylabel('position')
plt.legend()
plt.savefig('data.png',dpi=300)
plt.show()

x0=np.random.randn(3,1)
P0=100*np.eye(3,3)
R=0.5*np.eye(1,1)

# create a recursive least squares object
RLS=RecursiveLeastSquares(x0,P0,R)

# simulate online prediction
for j in np.arange(np.size(simulationTime)):
    C=np.array([[1,simulationTime[j],(simulationTime[j]**2)/2]])
    RLS.predict(positionNoisy[j],C)

# extract the estimates in order to plot the results
estimate1=[]
estimate2=[]
estimate3=[]    
for j in np.arange(np.size(simulationTime)):
    estimate1.append(RLS.estimates[j][0])
    estimate2.append(RLS.estimates[j][1])
    estimate3.append(RLS.estimates[j][2])
    
# create vectors corresponding to the true values in order to plot the results
estimate1true=initialPosition*np.ones(np.size(simulationTime))
estimate2true=initialVelocity*np.ones(np.size(simulationTime))
estimate3true=acceleration*np.ones(np.size(simulationTime))


# plot the results
steps=np.arange(np.size(simulationTime))
fig, ax = plt.subplots(3,1,figsize=(10,15))
ax[0].plot(steps,estimate1true,color='red',linestyle='-',linewidth=6,label='True value of position')
ax[0].plot(steps,estimate1,color='blue',linestyle='-',linewidth=3,label='True value of position')
ax[0].set_xlabel("Discrete-time steps k",fontsize=14)
ax[0].set_ylabel("Position",fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[0].set_ylim(98,102)  
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps,estimate2true,color='red',linestyle='-',linewidth=6,label='True value of velocity')
ax[1].plot(steps,estimate2,color='blue',linestyle='-',linewidth=3,label='Estimate of velocity')
ax[1].set_xlabel("Discrete-time steps k",fontsize=14)
ax[1].set_ylabel("Velocity",fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[1].grid()
ax[1].legend(fontsize=14)

ax[2].plot(steps,estimate3true,color='red',linestyle='-',linewidth=6,label='True value of acceleration')
ax[2].plot(steps,estimate3,color='blue',linestyle='-',linewidth=3,label='Estimate of acceleration')
ax[2].set_xlabel("Discrete-time steps k",fontsize=14)
ax[2].set_ylabel("Acceleration",fontsize=14)
ax[2].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[2].grid()
ax[2].legend(fontsize=14)

fig.savefig('plots.png',dpi=600)




