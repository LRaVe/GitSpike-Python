# -*- coding: utf-8 -*-
"""
Created on Tue July 14 2026
@author: Laure WOLFF (PySpike Native Version)
"""

import numpy as np
import matplotlib.pyplot as plt
import pyspike as spk

def SPIKE_Distance_matrix(CellMatrix, num_neurons, S, R, tmin, tmax, plotting):
    """
    Computes the pairwise SPIKE-distance matrices for all neurons using 
    PySpike's highly optimized native C-backend functions.
    
    Parameters:
    -----------
    CellMatrix : 3D list [neuron][stimulus][repetition] of pyspike.SpikeTrain
    num_neurons : int
    S : int (Number of stimuli)
    R : int (Number of repetitions)
    tmin, tmax : float (Time window bounds)
    plotting : bool (Whether to display the grid of matrices)
    
    Returns:
    --------
    All_Matrix_D : np.ndarray of shape (num_trials, num_trials, num_neurons)
    """
    num_trials = S * R
    All_Matrix_D = np.zeros((num_trials, num_trials, num_neurons))
    
    for n in range(num_neurons):
        precomputed_trains = []
        for st in range(S):
            for rp in range(R):
                raw_spikes = CellMatrix[n, st, rp]
                
                # Correction to read correctly the LL.mat's data
                if isinstance(raw_spikes, np.ndarray):
                    raw_spikes = raw_spikes.ravel() # Convertit [[0.008, ...]] en [0.008, ...]
                            
                # Convertion in SpikeTrain of Pyspike libairy
                # On passe le tableau de spikes, et l'intervalle de temps [tmin, tmax]
                spike_train = spk.SpikeTrain(raw_spikes, edges=(tmin, tmax))
                
                precomputed_trains.append(spike_train)
                # spike_train = CellMatrix[n][st][rp]
                # precomputed_trains.append(spike_train)
    
            matrix_d = spk.spike_distance_matrix(precomputed_trains, interval=(tmin, tmax))
            
        All_Matrix_D[:, :, n] = matrix_d

    # =========================================================================
    #  PLOTTING SECTION
    # =========================================================================
    if plotting:
        # Creation of the labels (ex: S1-R1, S1-R2...)
        trial_labels = []
        for st in range(1, S + 1):
            for rp in range(1, R + 1):
                trial_labels.append(f"S{st}-R{rp}")
                
        cols = int(np.ceil(np.sqrt(num_neurons * 1.25)))
        rows = int(np.ceil(num_neurons / cols))
        
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8), facecolor='w')
        fig.canvas.manager.set_window_title('Pairwise SPIKE Distance Matrices per Neuron')
        axs = np.atleast_1d(axs).flatten()
        
        for n in range(num_neurons):
            ax = axs[n]
            mat_to_plot = All_Matrix_D[:, :, n]
            
            im = ax.imshow(mat_to_plot, cmap='jet', origin='upper', aspect='equal')
            
            max_val = np.max(mat_to_plot)
            if max_val > 0:
                im.set_clim(0, max_val)
            else:
                im.set_clim(0, 1)
                
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Manage the scales
            if num_trials <= 12:
                ax.set_xticks(range(num_trials))
                ax.set_yticks(range(num_trials))
                ax.set_xticklabels(trial_labels, fontsize=6, rotation=45, ha='right')
                ax.set_yticklabels(trial_labels, fontsize=6)
            else:
                ax.set_xticks([0, num_trials - 1])
                ax.set_yticks([0, num_trials - 1])
                ax.set_xticklabels(['1', str(num_trials)], fontsize=8)
                ax.set_yticklabels(['1', str(num_trials)], fontsize=8)
                
            ax.set_title(f"Neuron {n+1}", fontsize=9, fontweight='bold')
            ax.set_xlabel('Trials', fontsize=7)
            ax.set_ylabel('Trials', fontsize=7)
            
        for empty_ax in axs[num_neurons:]:
            empty_ax.set_visible(False)
            
        plt.tight_layout()
        plt.show()

    return All_Matrix_D