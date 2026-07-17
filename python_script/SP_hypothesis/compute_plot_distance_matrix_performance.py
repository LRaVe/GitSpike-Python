# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:56:23 2026
@author: Laure WOLFF
Script to compute and plot the performance value and Spike distance matrix  
(figure 2 of the 2018's paper')
"""

import numpy as np
import matplotlib.pyplot as plt
from pyspike import *

def compute_distance_matrix_performance(cell_matrix,selection_mask, num_stimuli, num_repetitions,t_start,t_end):
    """
    Computes the distance matrix and performance value P for neurons selected by the mask,
    following the SP hypothesis (combining spike trains first).
    """
    num_trials = num_stimuli * num_repetitions
    matrix_d = np.zeros((num_trials, num_trials))
    
    idx_selected = np.where(np.array(selection_mask) == 1)[0]
    
    if len(idx_selected) == 0:
        return -np.inf, matrix_d
        
    for t_a in range(num_trials):
        st_a, rp_a = t_a // num_repetitions, t_a % num_repetitions
        for t_b in range(t_a + 1, num_trials):
            st_b, rp_b = t_b // num_repetitions, t_b % num_repetitions
            
            # To concatenate all the spikes for the SP hypothesis
            times_A = np.sort(np.concatenate([cell_matrix[nc][st_a][rp_a].spikes for nc in idx_selected]))
            times_B = np.sort(np.concatenate([cell_matrix[nc][st_b][rp_b].spikes for nc in idx_selected]))
            combined_A = SpikeTrain(times_A, [t_start, t_end])
            combined_B = SpikeTrain(times_B, [t_start, t_end])
            
            matrix_d[t_a, t_b] = spike_distance(combined_A, combined_B)
            matrix_d[t_b, t_a] = matrix_d[t_a, t_b]
            
    # Calcul of the performance P
    sum_intra, count_intra = 0.0, 0
    sum_inter, count_inter = 0.0, 0
    for t_a in range(num_trials):
        st_a = t_a // num_repetitions
        for t_b in range(t_a + 1, num_trials):
            st_b = t_b // num_repetitions
            if st_a == st_b:
                sum_intra += matrix_d[t_a, t_b]
                count_intra += 1
            else:
                sum_inter += matrix_d[t_a, t_b]
                count_inter += 1
                
    perf_P = (sum_inter / count_inter) - (sum_intra / count_intra) if count_intra > 0 else 0.0
    return perf_P, matrix_d

def plot_and_compute_distance_matrix(CellMatrix, num_neurons, num_coding_neurons, 
                                     num_stimuli, num_repetitions, t1, t2):
    """Computes and plots the distance matrices for Coding, Non-coding, and Full populations"""
    
    num_trials = num_stimuli * num_repetitions
    trial_labels = [f"S{st+1}-R{rp+1}" for st in range(num_stimuli) for rp in range(num_repetitions)]
    
    ## 2. Creation of the three selection masks
    coding_selection = np.concatenate([np.ones(num_coding_neurons), np.zeros(num_neurons - num_coding_neurons)])
    noise_selection  = np.concatenate([np.zeros(num_coding_neurons), np.ones(num_neurons - num_coding_neurons)])
    full_selection   = np.ones(num_neurons)
    
    perf_A, Matrix_A = compute_distance_matrix_performance(CellMatrix, coding_selection, num_stimuli, num_repetitions,t1,t2)
    perf_B, Matrix_B = compute_distance_matrix_performance(CellMatrix, noise_selection, num_stimuli, num_repetitions,t1,t2)
    perf_C, Matrix_C = compute_distance_matrix_performance(CellMatrix, full_selection, num_stimuli, num_repetitions,t1,t2)
    
    ## 4. Plotting the three distance matrices 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='w')
    fig.canvas.manager.set_window_title('SP Distances matrix')
    
    matrices = [Matrix_A, Matrix_B, Matrix_C]
    perfs = [perf_A, perf_B, perf_C]
    titles = ['A. Coding subpopulation (C)', 'B. Non-coding subpopulation (NC)', 'C. Full population (All)']
    colors = ['r', 'b', 'k']
    
    for i, ax in enumerate(axes):
        im = ax.imshow(matrices[i], cmap='jet', origin='upper')
        
        # Axis confuguration
        ax.set_xticks(range(num_trials))
        ax.set_yticks(range(num_trials))
        ax.set_xticklabels(trial_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(trial_labels, fontsize=8)
        
        # Title configuration
        ax.set_title(f"{titles[i]}\nP = {perfs[i]:.4f}", color=colors[i], fontweight='bold', fontsize=10)
        ax.set_xlabel('Trials')
        
        if i == 0:
            ax.set_ylabel('Trials')
            
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.show()