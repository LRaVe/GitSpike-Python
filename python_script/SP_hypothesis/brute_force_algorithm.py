# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 2026
@author: Laure WOLFF
Brute Force (Exhaustive Search) Algorithm by Binary Incrementation
"""

import numpy as np
import matplotlib.pyplot as plt
from compute_plot_distance_matrix_performance import compute_distance_matrix_performance

def f_brute_force(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, other_figs):
    
    # Total number of possible combinations (2^N - 1, ignoring the all-zero mask)
    total_combinations = (2 ** num_neurons) - 1
    
    # # Safety check to prevent computer freezing if N is set too high
    # if num_neurons > 20:
    #     raise ValueError(f"Brute Force aborted: N is too large ({num_neurons}). Reduce N between 10 and 20 in your main script.")
        
    if showing:
        print(f"-> Launching Brute Force by binary incrementation ({total_combinations} masks to evaluate...)")
        
    # Variables initialization
    best_perf_overall = -np.inf
    best_mask_overall = np.zeros(num_neurons)
    history_perf_brute = np.zeros(total_combinations)
    
    # =====================================================================
    # STANDARD SEQUENTIAL LOOP WITH FAST BINARY EXTRACTION
    # =====================================================================
    for i in range(1, total_combinations + 1):
        current_mask = np.array([(i >> j) & 1 for j in range(num_neurons)])
        perf, _ = compute_distance_matrix_performance(
            CellMatrix, current_mask, num_stimuli, num_repetitions, t1, t2
        )
        
        history_perf_brute[i - 1] = perf
        
        if perf > best_perf_overall:
            best_perf_overall = perf
            best_mask_overall = current_mask.copy()
            
    best_subpop = np.where(best_mask_overall == 1)[0] + 1
    
    if showing:
        print("\n================ BRUTE FORCE CONVERGED ================")
        print(f"Best binary combination found: {best_subpop}")
        print(f"Absolute maximum performance P = {best_perf_overall:.4f}")
        print("=======================================================")
        
    ## Plotting the performance evolution
    if other_figs and len(history_perf_brute) > 0:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='w')
        fig.canvas.manager.set_window_title('Brute Force - Combinatorial Search History')
        
        iterations = np.arange(1, total_combinations + 1)
        
        # Plot every tested combination's performance
        ax.plot(iterations, history_perf_brute, color=[0.5, 0.5, 0.5], linewidth=0.8, label='Evaluated Mask Performance')
        
        # Reconstruct and plot the step-by-step maximum progress line (cummax)
        best_so_far = np.maximum.accumulate(history_perf_brute)
        ax.plot(iterations, best_so_far, 'b-', linewidth=2, label='Global Maximum Progress')
        
        # Highlight the global maximum point
        idx_max = np.where(history_perf_brute == best_perf_overall)[0][0]
        ax.plot(idx_max + 1, best_perf_overall, 'ro', markersize=8, markerfacecolor=[1, 0.2, 0.2], label='Absolute Best Solution')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(1, total_combinations)
        
        # Adjust Y limits based on data dynamically
        min_p = max(0.0, np.min(history_perf_brute))
        max_p = np.max(history_perf_brute)
        ax.set_ylim(min_p, max(max_p * 1.1, 0.1))
        
        ax.set_xlabel('Binary Counter Iterations (Search Space)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance P', fontsize=12, fontweight='bold')
        ax.set_title(f"Brute Force Search Tree Exploration (N = {num_neurons} Neurons)", fontsize=13, fontweight='bold')
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
    return best_subpop, best_perf_overall