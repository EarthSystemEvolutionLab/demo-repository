#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:11:25 2018

@author: benjohnson
"""
import numpy as np
import scipy as sp
import scipy.misc
from scipy.integrate import odeint
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 16}

### ADDED LINE FOR TESTING BRANCHING AND SUCH ####

## ADDED ANOTHER LINE ON A FORK IN MY OWN REPOSITORY ###

##DOING MORE STUFF IN THIS FORK IN MY OWN REPO###

##ANOTHER TEST ON MY OWN REPO##

##NOW ADDING A LINE IN THE ORGANIZATION REPO, TO PULL TO PERSONAL ONE###

matplotlib.rc('font', **font)
#%% ------ Script after Gregory, 1991 to test the steady state of ocean d18O values 
#          based on changing a number of different exchange rates, and allowing high
#          and low T seafloor alteration to be distinct fluxes

t = 4.4 #time in Gyr
Wo = 7 #original seawater d18O
W_SS = -1 #steady state 
num_steps = 40 #num of initial model steps

# rate constants in Gyr-1, from Muehlenbachs, 1998
k_weath = 8 #8 continental weathering
k_growth = 1.2 #continental growth nominal 1.2
k_hiT = 14.6 #14.6 high temperature seafloor 
k_loT = 1.7 #low temp seafloor/seafloor weathering 
k_W_recycling = 0.6 #water recycling at subduction zones

#fractionations (permil) btwn rock and water, from Muehlenbachs, 1998 except weathering, which we tuned to reproduce -1permil ocean

#Delt_weath = 12*np.ones(num_steps) # np.linspace(1,12,num=num_steps) # #nominal= 12
#Delt_growth = 8*np.ones(num_steps) #  np.linspace(1,10,num=num_steps)  #nominal = 10
#Delt_hiT = 1.3 #
#Delt_lowT =  8.6 #9.6  #basalt to clay 
#Delt_water_recycling = 2.5 #

Delt_weath = 19*np.ones(num_steps) #19
Delt_growth =  9.8*np.ones(num_steps)#10
Delt_hiT_mid =  1.5 #meuh = 4.1, Johnson and Wing 2020 1.5
Delt_hiT = 4.1 
Delt_lowT =  9.6 #9.6  #basalt to clay 
Delt_water_recycling = 3.5 # 3.5

delW = (Wo-W_SS)*np.exp(1)-np.sum([k_weath*t,k_hiT*t,k_loT*t])+W_SS

del_steady = np.sum([k_weath*(Wo-Delt_weath),k_growth*(Wo-Delt_growth),k_hiT*(Wo-Delt_hiT),k_loT*(Wo-Delt_lowT),k_W_recycling*(Wo-Delt_water_recycling)])\
            /np.sum([k_weath,k_growth,k_hiT,k_loT,k_W_recycling])
print(['Steady state =' ,del_steady])

#%% ---- Changing over time, change cont growth and weathering and low-temp....
#  N(t)=N_{0}e^{-\lambda t},}
#calculate steady state in 250 myr increments 
del_graniteo = np.linspace(7.8,7.8,num=num_steps)   #7.8
del_basalto = 5.8
del_WR = 7
bb= 0.3
time = np.linspace(0,4.5,num=num_steps) #sample every 250 myr
weath_time_on = 4.5-2.2#in Ga
weath_time_early = 4.5-4.43
weath_time_late = 4.5-0.9

Delt_hiT_change = (Delt_hiT-Delt_hiT_mid)+(Delt_hiT-Delt_hiT_mid)*0.5*(1+np.tanh((np.subtract(time,weath_time_on)/bb))) -1
Delt_hiT_change_late = (Delt_hiT-Delt_hiT_mid)+(Delt_hiT-Delt_hiT_mid)*0.5*(1+np.tanh((np.subtract(time,weath_time_late)/bb))) -1
k_growth_change = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_on)/bb)))
k_growth_late = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_late)/bb)))
k_growth_early = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_early)/bb)))
#k_growth_change = k_growth*np.ones(time.size)
k_growth_change = (1-np.exp(-time))+k_growth
k_weathering_change =0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_on)/bb)))
k_weathering_late =0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_late)/bb)))
k_weathering_early =0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_early)/bb)))
#k_weathering_change = k_weath*np.ones(time.size)
#k_loT_change = 0.2*k_loT*(1-np.exp(-time)) #change low T alteration
k_loT_change = k_loT*np.ones(time.size) #keep it the same
#k_loT_change = np.linspace(10*k_loT,k_loT,num=num_steps)
k_hiT_change = k_hiT*np.ones(time.size)
#k_hiT_change = 0.1*k_hiT*(np.exp(-time))
#k_hiT_change = np.linspace(0.75*k_hiT,k_hiT,num=num_steps)
k_water_change = k_W_recycling*np.ones(time.size) #
#k_water_change = 10*k_W_recycling*np.exp(-time)

del_steady_change = np.zeros(time.size)
del_steady_early = np.zeros(time.size)
del_steady_late = np.zeros(time.size)
k_sum = np.zeros(time.size)
k_sum_early = np.zeros(time.size)
k_sum_late = np.zeros(time.size)

for istep in range(0,time.size):
    top = np.sum([k_weathering_change[istep]*(del_graniteo[istep]-Delt_weath[istep]),\
                  k_growth_change[istep]*(del_graniteo[istep]-Delt_growth[istep]),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_change_late[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    top_early = np.sum([k_weathering_early[istep]*(del_graniteo[istep]-Delt_weath[istep]),\
                  k_growth_early[istep]*(del_graniteo[istep]-Delt_growth[istep]),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_change_late[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    top_late = np.sum([k_weathering_late[istep]*(del_graniteo[istep]-Delt_weath[istep]),\
                  k_growth_late[istep]*(del_graniteo[istep]-Delt_growth[istep]),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_change_late[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    k_sum[istep] = np.sum([k_weathering_change[istep],k_growth_change[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    k_sum_early[istep] = np.sum([k_weathering_early[istep],k_growth_early[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    k_sum_late[istep] = np.sum([k_weathering_late[istep],k_growth_late[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    
    del_steady_change[istep] = top/k_sum[istep]
    del_steady_early[istep] = top_early/k_sum_early[istep]
    del_steady_late[istep] = top_late/k_sum_late[istep]
       
#calculate dW at for each steady state
new_time_steps=1000    
time_new = np.linspace(0.01,4.5,num=new_time_steps)
f1 = interp1d(time,del_steady_change)
f2 = interp1d(time,k_sum)
steady_interp = f1(time_new)
k_sum_interp = f2(time_new)
f1_late = interp1d(time,del_steady_late)
f2_late = interp1d(time,k_sum_late)
steady_interp_late = f1_late(time_new)
k_sum_interp_late = f2_late(time_new)

steady_interp_late = f1_late(time_new)
k_sum_interp_late = f2_late(time_new)

f1_early = interp1d(time,del_steady_early)
f2_early = interp1d(time,k_sum_early)
steady_interp_early = f1_early(time_new)
k_sum_interp_early = f2_early(time_new)

dW_middle = np.add(np.subtract(Wo,steady_interp)*np.exp(-np.multiply(time_new,k_sum_interp)),steady_interp) 
dW_early = np.add(np.subtract(Wo,steady_interp_early)*np.exp(-np.multiply(time_new,k_sum_interp_early)),steady_interp_early) 
dW_late = np.add(np.subtract(Wo,steady_interp_late)*np.exp(-np.multiply(time_new,k_sum_interp_late)),steady_interp_late) 
dW_superlate = np.add(np.subtract(Wo,steady_interp_late)*np.exp(-np.multiply(time_new,k_sum_interp_late)),steady_interp_late) 

flux_factor=np.linspace(0.05,0.05,num=time_new.size)
dW_decay = np.add(np.subtract(Wo,steady_interp[-1])*np.exp(-np.multiply(time_new,flux_factor*k_sum[-1])),steady_interp[-1])


#solve numerically
flux_labels = ['Weath','Growth','hiT','loT','recy']
Deltas = pd.Series([Delt_weath,Delt_growth,Delt_hiT,Delt_lowT,Delt_water_recycling],index=flux_labels)
ks = pd.Series([k_weath,k_growth,k_hiT,k_loT,k_W_recycling],index=flux_labels)

def dW_dt(W,t,Delta,k,weath_time,flux_labels):
    kweatheringT = 0.5*k['Weath']*(1+np.tanh((np.subtract(t,weath_time)/1e-6)))
    kgrowthT = 0.5*k['Growth']*(1+np.tanh((np.subtract(t,weath_time)/1e-6)))
    ks['Weath'] = kweatheringT
    ks['Growth'] = kgrowthT
    dWdt = (-(np.multiply(k['Weath'],(W+Delta['Weath']- Wo))))\
            +(-(np.multiply(k['Growth'],(W+Delta['Growth']- Wo))))\
            +(-(np.multiply(k['hiT'],(W+Delta['hiT']- Wo))))\
            +(-(np.multiply(k['loT'],(W+Delta['loT']- Wo))))\
            +(-(np.multiply(k['recy'],(W+Delta['recy']- Wo))))
    return dWdt

#def dW_dt(dW,dtime,Delta,k):
#
#    dWsum = (-np.multiply(k,(dW+Deltas-Wo)))
#    return dWsum

dtime = time_new
weath_time = weath_time_on

#dW_test = odeint(dW_dt,Wo,dtime,args=(Delt_weath,k_weath))
# W_init = 56
# dW_dynamic = odeint(dW_dt,dtime,Wo,args = (Deltas,ks,weath_time,flux_labels))
# dW_dynamic = np.array(dW_dynamic).flatten()

#calculate dW at for each steady state
#dW = np.subtract(Wo,del_steady_change)*np.exp(-np.sum([k_growth_change*time])

#%% 
#plots 
time_labels = ['4.5','4','3.5','3','2.5','2','1.5','1','0.5','0']
time_ticks = np.linspace(0,4.5,10)
plt.close('all')
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1 = plt.subplot(1,1,1)
ax1.plot(time,k_growth_change,'k:')
ax1.plot(time,k_weathering_change,'k--')
ax1.plot(time,k_loT_change,'k-.')
ax1.plot(time,k_hiT_change,'k-')
ax1.plot(time,k_water_change,'k.-')
ax1.legend(['Continental recycling','Continental weathering','low T alteration','high T alteration','Water recycling']\
           ,bbox_to_anchor=(0.01, 0.93), loc=2, borderaxespad=0.,fontsize=8)
plt.xlabel('Age (Ga)'); plt.ylabel('Ocean $\delta^{18}$O')
locs, labels = plt.xticks()           # Get locations and labels

plt.xticks(time_ticks, time_labels)  # Set locations and labels
plt.xticks(time_ticks, time_labels)  
ax1.set_xlim([0,4.5])
plt.ylabel('Rate (Gyr $^{-1}$)')

fig2, (ax2) = plt.subplots(1, 1, sharey=True)

ax2 = plt.subplot(1,1,1) # subplot(2,1,2) is now active
#ax2.set_xlim([0,4.5])
plt.xticks(time_ticks, time_labels)  # Set locations and labels
plt.xlabel('Age (Ga)'); plt.ylabel('Ocean $\delta^{18}$O')
#plt.yticks([])
#plt.title('Ocean $\delta^{18}$O constraints')
ax2.plot(time_new,dW_decay,'k--',linewidth=2)
ax2.plot(time_new,dW_early,'k-.',linewidth=2)
ax2.plot(time_new,dW_middle,'k:',linewidth=2)
ax2.plot(time_new,dW_late,'k-',linewidth=2)
#ax2.plot(time_new,dW_dynamic,'g^')

#ax2.set_ylim([-3,7])
#ax2.set_aspect(0.2)
ax2.legend(['5% rate', 'Early continent emergence','Middle continent emergence','Late continent emergence'])#,'Early emergence','Late emergence'])
other_color = np.divide([219,168,133],255)
our_color = np.divide([144,110,110],255)
Pope = patches.Rectangle((4.5-3.75,0.8),0.1,3,linewidth=1,edgecolor='k',facecolor=other_color)
ax2.add_patch(Pope)
Ours = patches.Rectangle((4.5-3.19,2.3),0.1,0.62,linewidth=1,edgecolor='k',facecolor=our_color)
ax2.add_patch(Ours)
#−1.33 ± 0.98‰
Hodel = patches.Rectangle((4.5-0.71,-2.28),0.1,1.95,linewidth=1,edgecolor='k',facecolor=other_color)
ax2.add_patch(Hodel)
#0+-2
Ordo = patches.Rectangle((4.5-0.5,-2),0.1,4,linewidth=1,edgecolor='k',facecolor=other_color) 
ax2.add_patch(Ordo)
Samail =patches.Rectangle((4.5-0.085,-1.4),0.1,2,linewidth=1,edgecolor='k',facecolor=other_color) 
ax2.add_patch(Samail)

mag_oc = patches.Rectangle((0,6),0.1,2,linewidth=1,edgecolor='k',facecolor=other_color) 
ax2.add_patch(mag_oc)

#a_em =patches.Rectangle((4.5-2.5,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
#ax2.add_patch(a_em)
#b_em =patches.Rectangle((4.5-0.7,-3),.13,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
#ax2.add_patch(b_em)
#c_em =patches.Rectangle((4.5-3.5,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
#ax2.add_patch(c_em)
#d_em =patches.Rectangle((4.5-3.2,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
#ax2.add_patch(d_em)
#e_em =patches.Rectangle((4.5-3,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
#ax2.add_patch(e_em)
#plt.ylim([-2.5,7])
plt.xlim([0,4.5])
#
#qbox = patches.Rectangle((4.5-3.2,-1),2.5,2,linewidth=1,edgecolor='k',facecolor='xkcd:chartreuse',alpha=0.4) 
#ax2.add_patch(qbox)
#plt.text(4.5-2.4,-0.1,'???????',color='k',fontsize=20)

fig2.set_size_inches(12, 8.5)

#Oxygen
