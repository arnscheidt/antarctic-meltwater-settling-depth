#!/usr/bin/env python3
# Solves the simple line plume model numerically, and reproduces the corresponding figure from the paper
# Constantin Arnscheidt, 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode,odeint

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')

F = 10
L = 1000
N = 0.003
k = 0.001 

FLvec = np.array([5,10,15])/1000
alpha = 0.15

def model(t,X,params): 

    alpha, N, k = params

    Q, M, B, A = X

    Qdot = alpha * M/Q
    Mdot = Q*B/M
    Bdot = -(N**2)*Q
    Adot = -k*A*Mdot/Qdot
    return [Qdot, Mdot, Bdot, Adot]

z = np.arange(0,400,1)
dz = z[1]-z[0]

f1 = np.zeros((len(z),4,len(FLvec)))
lnb_i = np.zeros((len(FLvec)))

print(f1.shape)
for ifl in range(0,len(FLvec)):
    r = ode(model).set_integrator("dop853")
    r.set_initial_value((0.000001,0.000001,FLvec[ifl],1)).set_f_params([alpha,N,k])

    for j in range(0,len(z)):
        r.set_f_params([alpha,N,k])
        r.integrate(r.t+dz)
        f1[j,:,ifl] = r.y
    
    lnb_i[ifl] = np.argmin((f1[:,2,ifl]**2))
    

scatsize=70

lw = 3
fig = plt.figure()
plt.subplot(131)
plt.plot(f1[:,0,:],z,linewidth = lw)
plt.xlabel(r'Q (Volume flux, m$^2$/s)')
plt.ylabel('h (m)')
plt.scatter([f1[int(lnb_i[i]),0,i] for i in range(0,3)],[z[i] for i in lnb_i.astype(int)],color=(0,0,0),zorder=10,s=scatsize)
plt.legend([r'$F/L = $ '+str(i)+' m$^3$/s$^3$' for i in FLvec],framealpha=1,loc="upper left",fontsize=12)

ax2 = plt.subplot(132)
plt.plot(f1[:,1],z,linewidth = lw)
plt.xlabel('M (Momentum flux, m$^3$/s$^2$)')
ax2.get_yaxis().set_ticklabels([])
plt.scatter([f1[int(lnb_i[i]),1,i] for i in range(0,3)],[z[i] for i in lnb_i.astype(int)],color=(0,0,0),zorder=10,s=scatsize)

ax3 = plt.subplot(133)
plt.plot(f1[:,2],z,linewidth = lw)
plt.xlabel('B (Buoyancy flux, m$^3$/s$^3$)')
ax3.get_yaxis().set_ticklabels([])
plt.scatter([f1[int(lnb_i[i]),2,i] for i in range(0,3)],[z[i] for i in lnb_i.astype(int)],color=(0,0,0),zorder=10,s=scatsize)

plt.show()
