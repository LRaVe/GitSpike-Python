# -*- coding: utf-8 -*-
"""
ISI-distance computation with auxiliary boundary spikes and plotting 

Created on Thu Apr 30 14:08:16 2026

@author: laure WOLFF
"""
import matplotlib.pyplot as plt
import pyspike as spk
from numpy import *

# The testing dataset

tmin, tmax = 0, 10
num_trains = 3
spikes = [
    [0, 1, 2, 4, 7],
    [3, 4, 6, 10],
    [2, 5]
]

# dataset = 8

# if dataset == 8:
#    tmax=10;
#    tmin=0
#    spike_trains = []
#    spike_trains.append(spk.SpikeTrain([0,1.9, 3.9, 6,10], [tmin, tmax]))
#    spike_trains.append(spk.SpikeTrain([0,2, 6.1,10], [tmin, tmax]))
#    spike_trains.append(spk.SpikeTrain([0,2.1, 4.1, 6.2,10], [tmin, tmax]))
#    spike_trains.append(spk.SpikeTrain([0,2.2, 4.2, 5.9,10], [tmin, tmax]))
#    num_trains = len(spike_trains)

"""
The fonctions using to calculate the ISI-distance
"""

def f_spike_conform (spikes, tmin, tmax):
    n= len(spikes)
    for i in range (0,n):
        if spikes[i][0]!=tmin:
            spikes[i].insert(0,tmin)
        if spikes[i][-1]!=tmax:
            spikes[i].append(tmax)
    return spikes

def f_ISI_calculate (spikes):
    ISI_distance = []
    for i in range (len(spikes)):
        ISI_distance_i = []
        for j in range (len(spikes[i])-1):
            ISI_distance_i.append(spikes[i][j+1]-spikes[i][j])
        ISI_distance.append(ISI_distance_i)
    return ISI_distance

def I_t (value1, value2):
    return abs (value1 - value2) / max(value1, value2)

def I_A_t (spikes):
    I_A_t_list = []
    N = len(spikes)
    isi_values = f_ISI_calculate(spikes)
    for n in range (N):
        for m in range (n+1, N+1):
            I_A_t.append(I_t())
            
    return I_A_t_list

"""
The fonction to calculate the several ISI-distance and plotting the elements
"""
def f_ISI_distance (spikes, tmin, tmax):
     
     n = len (spikes)
     isi_values = f_ISI_calculate(spikes)
     
     # For ONE spike-trains
     if n == 1:
        plt.figure()
        y_plot = isi_values[0] + [isi_values[0][-1]]
        
        plt.step(spikes[0], y_plot, where='post')
        
        plt.title(f"Single Spike Train ISI Evolution")
        avg_isi = sum(isi_values[0]) / (tmax - tmin)
        plt.suptitle(f"Average ISI: {avg_isi:.4f}")
        
        plt.ylim(0, max(y_plot) * 1.1) 
        plt.show()
        return avg_isi
    
     #Initialzation for the plottings
     ISI_matrix = zeros((n, n))
     pair_data = [] # Pour stocker les évolutions temporelles
     all_t_events = [tmin, tmax]
    
     # Calculation of pairwise ISI_disatance
     for i in range(n):
         for j in range(i + 1, n):
            #Creatting the time axis
             t_all = sorted(list(set(spikes[i]) | set(spikes[j]) | {tmin, tmax}))
             all_t_events.extend(t_all)
            
             i_ij_acc = 0 
             it_list = [] # Times evolution
        
             for k in range(len(t_all) - 1):
                t_mid = (t_all[k] + t_all[k+1]) / 2.0
                
                ## Edge correction ############################################
                if not spikes[i] or t_mid < spikes[i][0]:
                    val_x = spikes[i][0] - tmin if spikes[i] else tmax - tmin
                elif t_mid > spikes[i][-1]:
                    val_x = tmax - spikes[i][-1]
                else:
                    # REsearch the index of the latest spike before t_mid
                    idx = len([s for s in spikes[i] if s <= t_mid]) - 1
                    val_x = spikes[i][idx+1] - spikes[i][idx]
                
                if not spikes[j] or t_mid < spikes[j][0]:
                    val_y = spikes[j][0] - tmin if spikes[j] else tmax - tmin
                elif t_mid > spikes[j][-1]:
                    val_y = tmax - spikes[j][-1]
                else:
                    idx = len([s for s in spikes[j] if s <= t_mid]) - 1
                    val_y = spikes[j][idx+1] - spikes[j][idx]
                
                current_i_t = I_t(val_x, val_y)
            
                i_ij_acc += current_i_t * (t_all[k+1] - t_all[k])
                it_list.append(current_i_t)
        
             dist_val = i_ij_acc / (tmax - tmin)
             ISI_matrix[i, j] = dist_val
             ISI_matrix[j, i] = dist_val
        
             pair_data.append({'t': t_all, 'It': it_list})
            
             plt.figure()
             plt.step(t_all, it_list + [it_list[-1]], where='post')
             plt.title(f"Pair {i+1} vs {j+1}")
             plt.suptitle(f"Pairwise ISI-distance: {dist_val:.4f}")            
             plt.ylim(0, 1)
             plt.margins(x=0, y=0)
             plt.show()
             
     t_global = unique(all_t_events)
     I_matrix = zeros((len(pair_data), len(t_global) - 1))
   
     for p in range(len(pair_data)):
         tp = pair_data[p]['t']
         itp = pair_data[p]['It']
         for k in range(len(t_global) - 1):
             t_mid = (t_global[k] + t_global[k+1]) / 2.0
             indices = where(tp[:-1] <= t_mid)[0]
             if indices.size > 0:
                idx = indices[-1] 
             else:
                idx = 0 
             I_matrix[p, k] = itp[idx]
           
     I_pop_mean = mean(I_matrix, axis=0)
     dt = diff(t_global) 
     global_isi_val = sum(I_pop_mean * dt) / (tmax - tmin)


     # Plot global
     plt.figure()
     plt.step(t_global, append(I_pop_mean, I_pop_mean[-1]), where='post', color='red')
     plt.title("Evolution of Population Average ISI distance")
     plt.suptitle(f"Global ISI-distance: {global_isi_val:.4f}")   
     plt.xlim(0, tmax)
     plt.ylim(0, 1)
     plt.show()
     
     plt.figure()
     plt.imshow(ISI_matrix, cmap='viridis', extent=[0.5, n+0.5, n+0.5, 0.5])
     plt.xticks(range(1, n + 1)); 
     plt.yticks(range(1, n + 1));
     plt.xlabel('Spike_trains');
     plt.ylabel('Spike_trains');
     plt.colorbar()
     plt.title("Matrix of the ISI-distance")
     plt.show()
    
     return ISI_matrix, pair_data, sorted(list(set(all_t_events)))
  
f_ISI_distance(spikes,0,10)   