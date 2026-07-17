# -*- coding: utf-8 -*-
"""
Created on Tue July 14 2026
@author: Laure WOLFF 
(Performance Analysis & Subpopulation Selection - Figures 4E, F, G)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_and_plot_performance_matrix(All_Matrices_M, All_Matrix_D, num_neurons, num_stimuli, num_repetitions, showing, plotting):
    """
    Compute the performance and mapping in the case of Labeled Line hypothesis
    """
    
    P_all_neurons = np.zeros((num_stimuli, num_stimuli, num_neurons))
    mask_intra_diag = ~np.eye(num_repetitions, dtype=bool)
    
    for n in range(num_neurons):
        MatrixM = All_Matrices_M[:, :, n]
        distance_matrix = All_Matrix_D[:, :, n]
        
        for st1 in range(num_stimuli):
            idx1 = np.arange(st1 * num_repetitions, (st1 + 1) * num_repetitions)
            for st2 in range(num_stimuli):
                if st1 == st2:
                    continue
                
                # if the Wilcoxon criteria is null, its perf is null too
                if MatrixM[st1, st2] == 0:
                    P_all_neurons[st1, st2, n] = 0.0
                    continue
                
                idx2 = np.arange(st2 * num_repetitions, (st2 + 1) * num_repetitions)
                
                mean_inter = np.mean(distance_matrix[np.ix_(idx1, idx2)])
                
                intra_1 = distance_matrix[np.ix_(idx1, idx1)]
                dist_intra_1 = np.mean(intra_1[mask_intra_diag])
                
                intra_2 = distance_matrix[np.ix_(idx2, idx2)]
                dist_intra_2 = np.mean(intra_2[mask_intra_diag])
                
                P_all_neurons[st1, st2, n] = mean_inter - (dist_intra_1 + dist_intra_2) / 2.0

    P_pop = np.zeros((num_stimuli, num_stimuli))
    M_max = np.zeros((num_stimuli, num_stimuli), dtype=int)
    
    for s in range(num_stimuli):
        for s_prime in range(num_stimuli):
            if s == s_prime:
                continue
                
            perf_profile = P_all_neurons[s, s_prime, :]
            is_valid_coder = All_Matrices_M[s, s_prime, :]
            
            valid_perf = np.where(is_valid_coder == 1, perf_profile, -999.0)
            
            max_perf = np.max(valid_perf)
            if max_perf > 0:
                best_neuron_idx = np.argmax(valid_perf)
                P_pop[s, s_prime] = max_perf
                M_max[s, s_prime] = best_neuron_idx + 1
                
    opt_LL = np.unique(M_max[M_max > 0])
    
    mask_distinct_pairs = ~np.eye(num_stimuli, dtype=bool)
    PLL_total = np.mean(P_pop[mask_distinct_pairs])

    if showing:
        print("\n" + "="*60)
        print("              LABELED LINE PERFORMANCE RESULTS             ")
        print("="*60)
        print(f"\n▶ Global Labeled Line Performance (P_LL): {PLL_total:.4f} ")
        print(f"▶ Optimized Subpopulation (opt_LL)     : {list(opt_LL)}")
        print("\n▶ Matrix M_max (Best Neuron Mapping per pair):")
        print(np.array2string(M_max, prefix="   "))
        print("="*60)

    # =========================================================================
    # --- PLOTTING SECTION ---
    # =========================================================================
    if plotting:
        stim_labels = [f"S{st+1}" for st in range(num_stimuli)]
        
        # --- Figure 4E : Individual performance matrix P_n ---
        cols = int(np.ceil(np.sqrt(num_neurons)))
        rows = int(np.ceil(num_neurons / cols))
        
        fig_e, axs_e = plt.subplots(rows, cols, figsize=(12, 8), facecolor='w')
        fig_e.canvas.manager.set_window_title('Figure 4E: Individual Performance Matrices Pn')
        axs_e = np.atleast_1d(axs_e).flatten()
        
        for n in range(num_neurons):
            ax = axs_e[n]
            im = ax.imshow(P_all_neurons[:, :, n], cmap='jet', origin='upper')
            fig_e.colorbar(im, ax=ax, shrink=0.75)
            ax.set_aspect('equal')
            ax.set_xticks(range(num_stimuli))
            ax.set_yticks(range(num_stimuli))
            ax.set_xticklabels(stim_labels, fontsize=8)
            ax.set_yticklabels(stim_labels, fontsize=8)
            ax.set_title(f"$P_{{{n+1}}}$ (Neuron {n+1})", fontsize=10, fontweight='bold')
            
        for empty_ax in axs_e[num_neurons:]:
            empty_ax.set_visible(False)
        plt.tight_layout()
        
        # --- Figures 4F & 4G : Global perforamnce and mapping matrices ---
        fig_fg, (ax_f, ax_g) = plt.subplots(1, 2, figsize=(14, 6), facecolor='w')
        fig_fg.canvas.manager.set_window_title('Figures 4F & 4G: Population Discrimination Analysis')
        
        # Subplot F : P_max (Performance of the population)
        im_f = ax_f.imshow(P_pop, cmap='jet', origin='upper')
        fig_fg.colorbar(im_f, ax=ax_f, shrink=0.8)
        ax_f.set_aspect('equal')
        ax_f.set_xticks(range(num_stimuli))
        ax_f.set_yticks(range(num_stimuli))
        ax_f.set_xticklabels(stim_labels)
        ax_f.set_yticklabels(stim_labels)
        ax_f.set_title(
            f"$P_{{max}}$ (Population Performance)\n"
            f"Global LL performance | $P_{{LL}}$ = {PLL_total:.4f}",
            fontweight='bold', 
            fontsize=12,
            pad=10  
        )
        
        # Subplot G : M_max (Best neuron for each pair)
        custom_colors = np.array([
            [0.0, 0.0, 0.0],    # 0 -> Black
            [0.0, 0.45, 1.0],   # 1 -> Blue
            [1.0, 0.0, 0.0],    # 2 -> Red
            [0.0, 0.65, 0.0],   # 3 -> Green
            [1.0, 0.85, 0.0]    # 4 -> Yellow
        ])
        
        if (num_neurons + 1) > len(custom_colors):
            extra_count = (num_neurons + 1) - len(custom_colors)
            extra_colors = plt.cm.tab10(np.linspace(0, 1, extra_count))[:, :3]
            custom_colors = np.vstack([custom_colors, extra_colors])
            
        custom_cmap = mcolors.ListedColormap(custom_colors[:num_neurons + 1, :])
        
        im_g = ax_g.imshow(M_max, cmap=custom_cmap, vmin=0, vmax=num_neurons, origin='upper')
        cb = fig_fg.colorbar(im_g, ax=ax_g, shrink=0.8, ticks=range(num_neurons + 1))
        cb.ax.set_yticklabels(range(num_neurons + 1))
        
        ax_g.set_aspect('equal')
        ax_g.set_xticks(range(num_stimuli))
        ax_g.set_yticks(range(num_stimuli))
        ax_g.set_xticklabels(stim_labels)
        ax_g.set_yticklabels(stim_labels)
        opt_str = ", ".join(map(str, opt_LL))
        ax_g.set_title(
            f"$M_{{max}}$ (Best Neuron Mapping)\n"
            f"Optimized Subpopulation $opt_{{LL}}$ = [{opt_str}]",
            fontweight='bold', 
            fontsize=12,
            pad=10
        )
        plt.tight_layout()
        plt.show()
    return P_all_neurons, P_pop, M_max, opt_LL, PLL_total