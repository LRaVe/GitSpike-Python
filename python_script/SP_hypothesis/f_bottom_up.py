# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 2026
@author: Laure WOLFF
Bottom-up greedy optimization algorithm (Sequential Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from compute_plot_distance_matrix_performance import compute_distance_matrix_performance

def f_bottom_up(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, plotting, other_figs):
    
    # Initialization variables
    best_order = np.zeros(num_neurons, dtype=int)
    neurons_dispo = list(range(num_neurons)) 
    history_perf = np.zeros(num_neurons)
    Matrix_Grid = np.full((num_neurons, num_neurons), np.nan)
    
    for k in range(num_neurons):
        num_dispo = len(neurons_dispo)
        current_step_perf = np.zeros(num_dispo)
        
        # Base mask construction from previously selected neurons
        selection = np.zeros(num_neurons)
        if k > 0:
            selection[best_order[0:k]] = 1
            
        for i, neuron_test in enumerate(neurons_dispo):
            local_selection = selection.copy()
            local_selection[neuron_test] = 1
            perf, _ = compute_distance_matrix_performance(
                CellMatrix, local_selection, num_stimuli, num_repetitions, t1, t2
            )
            current_step_perf[i] = perf
        
        best_idx = np.argmax(current_step_perf)
        best_perf_step = current_step_perf[best_idx]
        best_neurone_step = neurons_dispo[best_idx]
        
        for i, n in enumerate(neurons_dispo):
            Matrix_Grid[k, n] = current_step_perf[i]
            
        best_order[k] = best_neurone_step
        neurons_dispo.pop(best_idx)  
        history_perf[k] = best_perf_step
        
        if showing:
            print(f"Step k = {k+1} | Adding neuron : {best_neurone_step + 1} | Performance P = {best_perf_step:.4f}")
    # =====================================================================

    # Finding the best subpopulation
    idx_max_absolu = np.argmax(history_perf)
    best_subpop_idx = best_order[0:idx_max_absolu + 1]
    
    best_order_print = best_order + 1
    best_subpop_print = best_subpop_idx + 1
    
    if showing:
        print("\n--- Performances history ---")
        for i in range(0, num_neurons, 10):
            chunk = history_perf[i:i+10]
            print(f"   [{i+1}-{min(i+10, num_neurons)}] : " + " ".join(f"{val:.4f}" for val in chunk))
            
        print("\nOptimal neuron inclusion order:")
        for i in range(0, num_neurons, 10):
            print("   " + " ".join(str(x) for x in best_order_print[i:i+10]))
            
        print(f"\nThe best subpopulation found contains {len(best_subpop_print)} neurons:")
        for i in range(0, len(best_subpop_print), 10):
            print("   " + " ".join(str(x) for x in best_subpop_print[i:i+10]))
            
        print(f"\nThe best performance found is: {max(history_perf):.4f}")

    if plotting:
        tick_step = 1 if num_neurons <= 15 else (5 if num_neurons <= 40 else 10)
        max_P = np.max(history_perf)
        idx_max = np.argmax(history_perf)
        min_perf_val = np.nanmin(Matrix_Grid) - 0.02
        
        if other_figs:
            # 1. the evolution of the performance
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            fig1.canvas.manager.set_window_title('Bottom-Up optimization results')
            ax1.plot(range(1, num_neurons + 1), history_perf, '-o', linewidth=2.5, color=[0.30, 0.58, 0.20],
                     markeredgecolor=[0.30, 0.58, 0.20], markerfacecolor=[0.93, 0.69, 0.13], markersize=8, label='Performance P(k)')
            ax1.axvline(x=idx_max + 1, color=[0.85, 0.33, 0.1], linestyle='--', linewidth=1.5, label='Optimal size threshold')
            ax1.grid(True, linestyle='--')
            ax1.set_xlim(0.5, num_neurons + 0.5)
            ax1.set_xlabel('Neurons integrated sequentially (Step k)', fontweight='bold')
            ax1.set_ylabel('Global performance P', fontweight='bold')
            ax1.set_title('Evolution of performance using Bottom-Up selection', fontweight='bold')
            
            text_str = f"Optimal subpopulation:\nNeurons: {list(best_subpop_print)}\nMax P = {max_P:.4f}"
            ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=9, fontweight='bold',
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC'))
            ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            plt.tight_layout()

            # 2. the matrix
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            fig2.canvas.manager.set_window_title('Bottom-Up selection matrix')
            masked_grid = np.ma.masked_invalid(Matrix_Grid)
            im = ax2.imshow(masked_grid, cmap='jet', aspect='auto', origin='upper')
            fig2.colorbar(im, ax=ax2)
            
            for step in range(num_neurons):
                ax2.plot(best_order[step], step, 'kx', markersize=10, linewidth=2.5)
                
            scaled_perf = (history_perf - np.nanmin(Matrix_Grid)) / (np.nanmax(Matrix_Grid) - np.nanmin(Matrix_Grid)) * (num_neurons - 1)
            ax2.plot(scaled_perf, range(num_neurons), '-r', linewidth=2.5, label='Max performance P')
            ax2.set_xlabel('# Neuron Index', fontweight='bold')
            ax2.set_ylabel('Number of neurons (Step k)', fontweight='bold')
            ax2.set_xticks(range(0, num_neurons, tick_step))
            ax2.set_xticklabels(range(1, num_neurons + 1, tick_step))
            ax2.set_title('Bottom-Up selection matrix', fontweight='bold')
            plt.tight_layout()

        # 3. Figure matrix + functikon (figure of the paper)
        fig3 = plt.figure(figsize=(11, 5), facecolor='w')
        fig3.canvas.manager.set_window_title('Bottom-Up selection figure')
        
        Matrix_Paper = Matrix_Grid.copy()
        for k in range(num_neurons):
            past_neurons = best_order[0:k]
            Matrix_Paper[k, past_neurons] = min_perf_val
            Matrix_Paper[k, best_order[k]] = history_perf[k]

        ax_mat = plt.subplot2grid((1, 5), (0, 0), colspan=3)
        im_paper = ax_mat.imshow(Matrix_Paper, cmap='jet', aspect='auto', origin='lower', 
                                 vmin=min_perf_val, vmax=max_P + 0.02)
        
        for i in np.arange(-0.5, num_neurons, 1):
            ax_mat.axhline(i, color='white', alpha=0.2, linewidth=0.5)
            ax_mat.axvline(i, color='white', alpha=0.2, linewidth=0.5)

        for k in range(num_neurons):
            n_id = best_order[k]
            ax_mat.text(n_id, k, '✓', color='black', fontsize=11, ha='center', va='center', fontweight='bold')
            if k < num_neurons - 1:
                ax_mat.plot([n_id] * (num_neurons - 1 - k), range(k + 1, num_neurons), '.', color=[0.3, 0.3, 0.3], markersize=5)
                
        for n_id in best_subpop_idx:
            ax_mat.plot(n_id, idx_max, 'rx', markersize=10, linewidth=2)
            
        rect = plt.Rectangle((-0.45, idx_max - 0.45), num_neurons, 0.9, linewidth=2, edgecolor=[0.15, 0.62, 0.15], facecolor='none')
        ax_mat.add_patch(rect)
        
        ax_mat.set_xticks(range(0, num_neurons, tick_step))
        ax_mat.set_xticklabels(range(1, num_neurons + 1, tick_step))
        ax_mat.set_yticks(range(0, num_neurons, tick_step))
        ax_mat.set_yticklabels(range(1, num_neurons + 1, tick_step))
        ax_mat.set_xlabel('Neuron ID', fontsize=11, fontweight='bold')
        ax_mat.set_ylabel('Size of population (k)', fontsize=11, fontweight='bold')
        ax_mat.set_title('Bottom-Up algorithm matrix', fontsize=12, fontweight='bold')
        
        cb = fig3.colorbar(im_paper, ax=ax_mat, fraction=0.03, pad=0.04)
        cb.set_label('Global Performance P', fontsize=11, fontweight='bold')

        ax_perf = plt.subplot2grid((1, 5), (0, 3), colspan=2)
        ax_perf.plot(history_perf, range(num_neurons), '-ko', linewidth=2, markerfacecolor='black', markersize=5)
        ax_perf.plot(history_perf[idx_max], idx_max, 'ro', markersize=11, linewidth=2, markerfacecolor='white')
        ax_perf.plot(history_perf[idx_max], idx_max, 'rx', markersize=7, linewidth=1.5)
        
        ax_perf.grid(True, linestyle='--')
        ax_perf.set_ylim(-0.5, num_neurons - 0.5)
        ax_perf.set_yticks(range(0, num_neurons, tick_step))
        ax_perf.set_yticklabels([]) 
        ax_perf.set_xlim(min_perf_val + 0.02, max_P + 0.03)
        ax_perf.set_xlabel('Best performance P', fontsize=11, fontweight='bold')
        ax_perf.set_title('Performance function', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    return best_subpop_print, max_P