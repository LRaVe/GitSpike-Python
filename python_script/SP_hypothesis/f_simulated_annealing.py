# -*- coding: utf-8 -*-
"""
Created on Mon July 13 2026
@author: Laure WOLFF
Simulated Annealing optimization algorithm (Sequential Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from compute_plot_distance_matrix_performance import compute_distance_matrix_performance

def f_simulated_annealing(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, plotting, other_figs):
    
    # 1. Initialization of the variables
    cooling_factor = 0.9        
    alpha_threshold = 1e-5       
    iterations_per_temp = 5 * num_neurons 
    N0 = 50
    max_paliers_est = 200 
    
    Matrix_Grid = np.full((max_paliers_est, num_neurons), np.nan)
    history_perf = np.zeros(max_paliers_est) 
    
    # Generation of a random mask (0 or 1)
    mask_0 = np.random.randint(0, 2, num_neurons)
    if np.sum(mask_0) == 0: 
        mask_0[np.random.randint(num_neurons)] = 1
    if np.sum(mask_0) == num_neurons: 
        mask_0[np.random.randint(num_neurons)] = 0
        
    P_0, _ = compute_distance_matrix_performance(CellMatrix, mask_0, num_stimuli, num_repetitions, t1, t2)
    
    best_perf_overall = P_0
    best_mask_overall = mask_0.copy()
    temp_mask = mask_0.copy()
    temp_perf = P_0
    
    delta_down = np.zeros(N0)
    count = 0
    
    # 2. Finding T_0 (Initial temperature calculation)
    for n in range(N0):
        idx = np.random.randint(num_neurons)
        next_mask = temp_mask.copy()
        next_mask[idx] = 1 - temp_mask[idx]
        
        if np.sum(next_mask) == 0 or np.sum(next_mask) == num_neurons: 
            continue
            
        next_perf, _ = compute_distance_matrix_performance(CellMatrix, next_mask, num_stimuli, num_repetitions, t1, t2)
            
        if next_perf <= temp_perf:
            delta_down[count] = abs(next_perf - temp_perf)
            count += 1
            
        temp_perf = next_perf
        temp_mask = next_mask.copy()
        
    if count > 0:
        filled_deltas = delta_down[0:count]
        valid_deltas = filled_deltas[np.isfinite(filled_deltas) & (filled_deltas != 0)]
        
        if len(valid_deltas) > 0:
            mean_delta = np.mean(valid_deltas)
        else:
            mean_delta = 0.005
            if showing:
                print("  [Warning SA] All initial samples hit local traps. Default delta applied.")
    else:
        mean_delta = 0.005
        
    T_0 = - mean_delta / np.log(0.95) 
    
    if T_0 <= 1e-7 or np.isnan(T_0) or np.isinf(T_0):
        T_0 = 0.5
        if showing:
            print(f"  [Warning SA] Invalid T_0. Temperature forced to {T_0:.2f} to prevent crashes.")
            
    if showing: 
        print(f"T_0 found: {T_0:.6f}")
        
    max_iter_est = max_paliers_est * iterations_per_temp
    hist_iter_P = np.zeros(max_iter_est)
    hist_iter_bestP = np.zeros(max_iter_est)
    hist_iter_size = np.zeros(max_iter_est)
    hist_iter_temp = np.zeros(max_iter_est)
    
    # 3. Simulated Annealing Loop
    theta = T_0            
    unchanged_temp_cycles = 0
    palier_idx = 0
    nb_iterations = 0
    
    while theta > alpha_threshold:
        # Security dynamic resizing of grids if we exceed max_paliers_est
        if palier_idx >= Matrix_Grid.shape[0]:
            new_size = Matrix_Grid.shape[0] * 2
            
            expanded_grid = np.full((new_size, num_neurons), np.nan)
            expanded_grid[0:Matrix_Grid.shape[0], :] = Matrix_Grid
            Matrix_Grid = expanded_grid
            
            expanded_history = np.zeros(new_size)
            expanded_history[0:len(history_perf)] = history_perf
            history_perf = expanded_history
            
        if showing: 
            print(f"Temp: {theta:.6f} | Current P: {temp_perf:.4f}") 
            
        for iter_step in range(iterations_per_temp):
            active_count = np.sum(temp_mask)
            next_mask = temp_mask.copy()
            
            if active_count == 1:
                zero_indices = np.where(temp_mask == 0)[0]
                idx_explore = np.random.choice(zero_indices)
                next_mask[idx_explore] = 1
            elif active_count == num_neurons:
                one_indices = np.where(temp_mask == 1)[0]
                idx_explore = np.random.choice(one_indices)
                next_mask[idx_explore] = 0
            else:
                idx_explore = np.random.randint(num_neurons)
                next_mask[idx_explore] = 1 - temp_mask[idx_explore]
                
            next_perf, _ = compute_distance_matrix_performance(CellMatrix, next_mask, num_stimuli, num_repetitions, t1, t2)
            
            if next_perf > temp_perf:
                temp_mask = next_mask.copy()
                temp_perf = next_perf
            else:
                q = np.exp(-abs(next_perf - temp_perf) / theta) 
                if np.random.rand() < q:
                    temp_mask = next_mask.copy()
                    temp_perf = next_perf
                    
            if temp_perf > best_perf_overall:
                best_perf_overall = temp_perf
                best_mask_overall = temp_mask.copy()
                
            hist_iter_P[nb_iterations] = temp_perf
            hist_iter_bestP[nb_iterations] = best_perf_overall
            hist_iter_size[nb_iterations] = np.sum(temp_mask)
            hist_iter_temp[nb_iterations] = theta
            nb_iterations += 1
            
        Matrix_Grid[palier_idx, :] = temp_mask
        history_perf[palier_idx] = temp_perf
        
        # Check convergence (Stagnation over 2 cycles)
        if palier_idx >= 1 and abs(history_perf[palier_idx] - history_perf[palier_idx-1]) < 1e-6:
            unchanged_temp_cycles += 1
            if unchanged_temp_cycles >= 2:
                if showing:
                    print("Exit: Performance remained unchanged for 2 consecutive temperature cycles.")
                break 
        else:
            unchanged_temp_cycles = 0
            
        palier_idx += 1
        theta *= cooling_factor
        
    # Crop arrays to real length
    Matrix_Grid = Matrix_Grid[0:palier_idx, :]
    history_perf = history_perf[0:palier_idx]
    hist_iter_P = hist_iter_P[0:nb_iterations]
    hist_iter_bestP = hist_iter_bestP[0:nb_iterations]
    hist_iter_size = hist_iter_size[0:nb_iterations]
    hist_iter_temp = hist_iter_temp[0:nb_iterations]
    
    # 4. Final Wrap-up
    best_subpop = np.where(best_mask_overall == 1)[0]
    best_subpop_print = best_subpop + 1  # Convert to 1-based indexes for humans
    
    if showing:
        print(f"Optimal subpopulation found: {list(best_subpop_print)}")
        print(f"Max performance P = {best_perf_overall:.4f}")
        print(f"Number of iterations: {nb_iterations}")
        
    # 5. Plotting
    if plotting and Matrix_Grid.shape[0] > 0:
        num_paliers_reals = Matrix_Grid.shape[0]
        
        tick_step_X = 1 if num_neurons <= 15 else (5 if num_neurons <= 40 else 10)
        tick_step_Y = 1 if num_paliers_reals <= 20 else (5 if num_paliers_reals <= 60 else 10)
        
        # --- Figure 1: Selection History Matrix & P(temp) ---
        fig1 = plt.figure(figsize=(10, 5), facecolor='w')
        fig1.canvas.manager.set_window_title('Results - Simulated Annealing')
        
        # Subplot 1: Matrix Map (Columns 1 to 3 equivalent)
        ax_mat = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        # Custom binary map like MATLAB's custom mymap
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap([[0.2, 0.4, 0.8], [0.9, 0.2, 0.2]])
        
        im = ax_mat.imshow(Matrix_Grid, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
        
        # Plot markers on the final row for best neurons
        for n_id in best_subpop:
            ax_mat.plot(n_id, num_paliers_reals - 1, 'rx', markersize=10, linewidth=2)
            
        ax_mat.set_xticks(range(0, num_neurons, tick_step_X))
        ax_mat.set_xticklabels(range(1, num_neurons + 1, tick_step_X))
        ax_mat.set_yticks(range(0, num_paliers_reals, tick_step_Y))
        ax_mat.set_yticklabels(range(1, num_paliers_reals + 1, tick_step_Y))
        
        ax_mat.set_xlabel('Neuron ID', fontsize=11, fontweight='bold')
        ax_mat.set_ylabel('Temperature Steps (Cooling)', fontsize=11, fontweight='bold')
        ax_mat.set_title('Simulated Annealing - Selected Neurons History', fontsize=12, fontweight='bold')
        
        cb = fig1.colorbar(im, ax=ax_mat, fraction=0.03, pad=0.04)
        cb.set_ticks([0.25, 0.75])
        cb.set_ticklabels(['Deactivated (0)', 'Activated (1)'])
        
        # Subplot 2: Vertical Performance
        ax_perf = plt.subplot2grid((1, 4), (0, 3), colspan=1)
        ax_perf.plot(history_perf, range(num_paliers_reals), '-ko', linewidth=1.5, markerfacecolor='k', markersize=4)
        ax_perf.plot(best_perf_overall, num_paliers_reals - 1, 'ro', markersize=10, linewidth=2, markerfacecolor='w')
        ax_perf.plot(best_perf_overall, num_paliers_reals - 1, 'rx', markersize=6, linewidth=1.5)
        
        ax_perf.grid(True, linestyle='--')
        ax_perf.set_ylim(-0.5, num_paliers_reals - 0.5)
        ax_perf.set_yticks(range(0, num_paliers_reals, tick_step_Y))
        ax_perf.set_yticklabels([])
        ax_perf.set_xlabel('Performance P', fontsize=11, fontweight='bold')
        ax_perf.set_title('P(temp)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # --- Figure 2: Convergence Diagnostics ---
        if other_figs:
            fig2, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor='w')
            fig2.canvas.manager.set_window_title('Simulated Annealing - Convergence Diagnostics')
            
            # 1. Discrimination performance
            axs[0].plot(hist_iter_P, 'k', linewidth=1.2, label='Current')
            axs[0].plot(hist_iter_bestP, 'r', linewidth=2, label='Best')
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Performance P')
            axs[0].legend(loc='best')
            axs[0].set_title('Discrimination performance')
            axs[0].grid(True, linestyle='--')
            
            # 2. Population size
            axs[1].plot(hist_iter_size, 'b', linewidth=1.2)
            axs[1].axhline(len(best_subpop), color='r', linestyle='--', linewidth=1.5, label='Optimal found')
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Population size')
            axs[1].set_title('Subpopulation Size Track')
            axs[1].legend(loc='best')
            axs[1].grid(True, linestyle='--')
            
            # 3. Temperature cooling
            axs[2].semilogy(hist_iter_temp, 'm', linewidth=1.5)
            axs[2].set_xlabel('Iteration')
            axs[2].set_ylabel('Temperature')
            axs[2].set_title('Cooling Schedule (Log Scale)')
            axs[2].grid(True, linestyle='--')
            
            sorted_print_subpop = sorted(list(best_subpop_print))
            fig2.suptitle(f"Simulated Annealing Dynamics | Best P = {best_perf_overall:.4f} | Best Population = {sorted_print_subpop}",
                          fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
    return nb_iterations