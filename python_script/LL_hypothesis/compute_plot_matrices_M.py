# -*- coding: utf-8 -*-
"""
Created on Tue July 14 2026
@author: Laure WOLFF 
Script to compute et plot the M matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ranksums  # Import the function to teh Wilcoxon's tests

def calculate_plot_matrix_M(All_MatrixD, num_neurons, num_stimuli, num_repetitions, plotting=True):
    """
    Computes and plots the statistical discrimination matrices (M_n) for all neurons
    using the official SciPy Wilcoxon rank-sum test.
    
    Parameters:
    -----------
    All_MatrixD : np.ndarray of shape (num_trials, num_trials, num_neurons)
        The pairwise SPIKE-distance matrices.
    num_neurons : int
        Number of recorded neurons.
    num_stimuli : int
        Number of different stimuli.
    num_repetitions : int
        Number of repetitions per stimulus.
    plotting : bool
        Whether to display the grid of discrimination matrices.
        
    Returns:
    --------
    All_Matrices_M : np.ndarray of shape (num_stimuli, num_stimuli, num_neurons)
        The binary discrimination matrices (0 or 1).
    """
    num_trials = num_stimuli * num_repetitions
    All_Matrices_M = np.zeros((num_stimuli, num_stimuli, num_neurons))
    
    stimulus_idx = np.repeat(np.arange(num_stimuli), num_repetitions)
    alpha_val = 0.001  # alpha from the paper
    
    for n in range(num_neurons):
        MatrixD = np.copy(All_MatrixD[:, :, n])
        MatrixD[np.isnan(MatrixD)] = 1.0  
        
        MatrixM = np.zeros((num_stimuli, num_stimuli))
        
        for st1 in range(num_stimuli):
            for st2 in range(st1 + 1, num_stimuli):
                idx_st1 = np.where(stimulus_idx == st1)[0]
                idx_st2 = np.where(stimulus_idx == st2)[0]
                
                intra_1 = MatrixD[np.ix_(idx_st1, idx_st1)]
                intra_2 = MatrixD[np.ix_(idx_st2, idx_st2)]
                
                # Superior part of the matrix (without the 0 of the middle)
                dist_intra_1 = intra_1[np.triu_indices(len(idx_st1), k=1)]
                dist_intra_2 = intra_2[np.triu_indices(len(idx_st2), k=1)]
                
                
                dist_inter = MatrixD[np.ix_(idx_st1, idx_st2)].flatten()
                
                h1, h2, h3 = 0, 0, 0
                
                # --- Test 1 : Inter vs Intra 1 ---
                if dist_inter.size > 0 and dist_intra_1.size > 0:
                    _, p_val1 = ranksums(dist_inter, dist_intra_1)
                    if p_val1 < alpha_val:
                        h1 = 1
                
                # --- Test 2 : Inter vs Intra 2 ---
                if dist_inter.size > 0 and dist_intra_2.size > 0:
                    _, p_val2 = ranksums(dist_inter, dist_intra_2)
                    if p_val2 < alpha_val:
                        h2 = 1
                        
                # --- Test 3 : Intra 1 vs Intra 2 ---
                if dist_intra_1.size > 0 and dist_intra_2.size > 0:
                    _, p_val3 = ranksums(dist_intra_1, dist_intra_2)
                    if p_val3 < alpha_val:
                        h3 = 1
                
                # Wilcoxon criteria : Discriminated (1) if at least one test is positif
                if (h1 == 1) or (h2 == 1) or (h3 == 1):
                    MatrixM[st1, st2] = 1
                    MatrixM[st2, st1] = 1
                else:
                    MatrixM[st1, st2] = 0
                    MatrixM[st2, st1] = 0
                    
        All_Matrices_M[:, :, n] = MatrixM

    # =========================================================================
    # PLOTTING SECTION
    # =========================================================================
    if plotting:
        stim_labels = [f"S{st+1}" for st in range(num_stimuli)]
        
        cols = int(np.ceil(np.sqrt(num_neurons * 1.25)))
        rows = int(np.ceil(num_neurons / cols))
        
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8), facecolor='w')
        fig.canvas.manager.set_window_title('LL Discrimination Matrices Mn (SciPy)')
        
        axs = np.atleast_1d(axs).flatten()
        hues = np.linspace(0.6, 1.6, num_neurons + 1)[:-1]
        hues = np.mod(hues, 1.0)
        
        for n in range(num_neurons):
            ax = axs[n]
            current_rgb = mcolors.hsv_to_rgb([hues[n], 0.9, 0.95])
            
            if num_neurons >= 3:
                if n == 0: current_rgb = np.array([0.0, 0.45, 1.0])
                elif n == 1: current_rgb = np.array([1.0, 0.0, 0.0])
                elif n == 2: current_rgb = np.array([0.0, 0.65, 0.0])
            
            custom_cmap = mcolors.ListedColormap([[0, 0, 0], current_rgb])
            
            ax.imshow(All_Matrices_M[:, :, n], cmap=custom_cmap, vmin=0, vmax=1, origin='upper')
            
            ax.set_xticks(range(num_stimuli))
            ax.set_yticks(range(num_stimuli))
            ax.set_xticklabels(stim_labels, fontsize=8)
            ax.set_yticklabels(stim_labels, fontsize=8)
            
            ax.set_title(f"Matrix $M_{{{n+1}}}$", fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            
        for empty_ax in axs[num_neurons:]:
            empty_ax.set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    return All_Matrices_M