# -*- coding: utf-8 -*-
"""
Created on Mon July 13 2026
@author: Laure WOLFF
Module to generate a dataset in summed population and plot paper style rasters
"""
import numpy as np
import matplotlib.pyplot as plt
from pyspike import *

def local_f_poisson(length, rate, refrac_period):
    """Compute the Poisson's law to generate the dataset"""
    uniform = np.random.rand(length)
    return refrac_period - np.log(1.0 - uniform) / rate

def generate_and_plot_raster(num_stimuli, num_repetitions, num_indi, num_coll, 
                             num_neurons, t1, t2, base_rate, refrac, plotting, other_figs):
    
    # Input parameter consistency check
    if (num_indi + num_coll) > num_neurons:
        raise ValueError('Error: The sum of num_indi and num_coll cannot exceed the total num_neurons!')
        
    num_trials = num_stimuli * num_repetitions
    
    # 3D matrix : [neurone]x[stimulus]x[repetition]
    CellMatrix = [
        [[None for _ in range(num_repetitions)] for _ in range(num_stimuli)]
        for _ in range(num_neurons)
    ]
    
    ## 1. Generation of COLLECTIVE coding neurons (Channels: 0 to num_coll-1)
    for st in range(num_stimuli):
        pooled_rate = num_coll * base_rate * 0.9
        approx_spikes = int(round((t2 - t1) * pooled_rate * 3)) + 10
        intervals = local_f_poisson(approx_spikes, pooled_rate, refrac)
        spikes_pooled = np.cumsum(intervals)
        spikes_pooled = spikes_pooled[(spikes_pooled >= t1) & (spikes_pooled <= t2)]
        num_spikes = len(spikes_pooled)
        
        for rp in range(num_repetitions):
            if num_spikes > 0:
                shuffled_indices = np.random.permutation(num_spikes)
                for nc in range(num_coll):
                    idx_assigned = shuffled_indices[nc::num_coll]
                    chosen_spikes = np.sort(spikes_pooled[idx_assigned])
                    CellMatrix[nc][st][rp] = SpikeTrain(chosen_spikes, edges=[t1, t2])
            else:
                for nc in range(num_coll):
                    CellMatrix[nc][st][rp] = SpikeTrain([], edges=[t1, t2])
                    
    ## 2. Generation of INDIVIDUAL coding neurons (Channels: num_coll to num_coll + num_indi - 1)
    for c_idx in range(num_indi):
        nc = num_coll + c_idx
        for st in range(num_stimuli):
            local_rate = base_rate * (0.75 + 1.0 * st)
            for rp in range(num_repetitions):
                approx_spikes = int(round((t2 - t1) * local_rate * 3)) + 10
                intervals = local_f_poisson(approx_spikes, local_rate, refrac)
                spikes = np.cumsum(intervals)
                chosen_spikes = spikes[(spikes >= t1) & (spikes <= t2)]
                CellMatrix[nc][st][rp] = SpikeTrain(chosen_spikes, edges=[t1, t2])
                
    ## 3. Generation of NON-CODING neurons
    for st in range(num_stimuli):
        for rp in range(num_repetitions):
            for nc in range(num_coll + num_indi, num_neurons):
                approx_spikes = int(round((t2 - t1) * base_rate * 1.0)) + 10
                intervals = local_f_poisson(approx_spikes, base_rate, refrac)
                spikes_noise = np.cumsum(intervals)
                chosen_spikes = spikes_noise[(spikes_noise >= t1) & (spikes_noise <= t2)]
                CellMatrix[nc][st][rp] = SpikeTrain(chosen_spikes, edges=[t1, t2])

    ## =========================================================================
    ## PLOTTING SECTION
    ## =========================================================================
    if plotting:
        trial_labels = [f"S{st+1}-R{rp+1}" for st in range(num_stimuli) for rp in range(num_repetitions)]
        color_coll = [0.85, 0.325, 0.098]  # Orange/Red
        color_indi = [1.0, 0.0, 0.0]       # Red
        color_noise = [0.0, 0.447, 0.741]  # Blue
        
        if other_figs:
            # --- FIG 1: Global Raster Plot ---
            fig, ax = plt.subplots(figsize=(9.5, 7), facecolor='w')
            ax.set_title('Artificial Dataset (Orange: Coll | Red: Indi | Blue: NC Noise)', fontsize=12, fontweight='bold')
            
            for t_idx in range(num_trials):
                st = t_idx // num_repetitions
                rp = t_idx % num_repetitions
                for nc in range(num_neurons):
                    spikes = CellMatrix[nc][st][rp].spikes
                    if len(spikes) > 0:
                        if nc < num_coll:
                            current_color, line_width = color_coll, 1.4
                        elif nc < (num_coll + num_indi):
                            current_color, line_width = color_indi, 1.4
                        else:
                            current_color, line_width = color_noise, 1.0
                        
                        ax.vlines(spikes, t_idx + 1 - 0.35, t_idx + 1 + 0.35, colors=current_color, linewidths=line_width)
                ax.axhline(t_idx + 1, color=[0.95, 0.95, 0.95], linewidth=0.5)
                
            for st_sep in range(1, num_stimuli):
                ax.axhline(st_sep * num_repetitions + 0.5, color=[0.2, 0.2, 0.2], linewidth=1.5, linestyle='--')
                
            ax.set_xlim(t1, t2)
            ax.set_ylim(0.5, num_trials + 0.5)
            ax.set_yticks(range(1, num_trials + 1))
            ax.set_yticklabels(trial_labels, fontsize=9)
            ax.set_xlabel('Time (au)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Trials (Stimuli / Repetitions)', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, which='both', axis='x', color='lightgray', linestyle='-', alpha=0.5)
            
        # --- FIG 2: Multi-Subplot layout per Trial ---
        fig2, axes2 = plt.subplots(num_trials, 1, figsize=(10, 8), sharex=True, facecolor='w')
        fig2.suptitle('Raster plot (Fig 1 of the 2018s paper)', fontsize=12, fontweight='bold')
        idx_subplot = 0
        
        for st_select in range(num_stimuli):
            for rp_select in range(num_repetitions):
                ax = axes2[idx_subplot]
                
                all_coll_spikes, all_indi_spikes, all_noise_spikes = [], [], []
                
                for nc in range(num_neurons):
                    spikes = CellMatrix[nc][st_select][rp_select].spikes
                    y_pos = (num_neurons - nc) + 4 
                    
                    if nc < num_coll:
                        current_color = color_coll
                        all_coll_spikes.extend(spikes)
                    elif nc < (num_coll + num_indi):
                        current_color = color_indi
                        all_indi_spikes.extend(spikes)
                    else:
                        current_color = color_noise
                        all_noise_spikes.extend(spikes)
                        
                    if len(spikes) > 0:
                        ax.vlines(spikes, y_pos - 0.35, y_pos + 0.35, colors=current_color, linewidths=0.8)
                
                all_coll_spikes = np.unique(np.sort(all_coll_spikes))
                all_indi_spikes = np.unique(np.sort(all_indi_spikes))
                all_noise_spikes = np.unique(np.sort(all_noise_spikes))
                all_total_spikes = np.unique(np.sort(np.concatenate([all_coll_spikes, all_indi_spikes, all_noise_spikes])))
                
                if len(all_total_spikes) > 0: ax.vlines(all_total_spikes, 1 - 0.4, 1 + 0.4, colors='black', linewidths=1.3)
                if len(all_noise_spikes) > 0: ax.vlines(all_noise_spikes, 2 - 0.3, 2 + 0.3, colors=color_noise, linewidths=1.0)
                if len(all_indi_spikes) > 0: ax.vlines(all_indi_spikes, 3 - 0.3, 3 + 0.3, colors=color_indi, linewidths=1.2)
                if len(all_coll_spikes) > 0: ax.vlines(all_coll_spikes, 4 - 0.3, 4 + 0.3, colors=color_coll, linewidths=1.2)
                
                ax.axhline(4.5, color=[0.3, 0.3, 0.3], linewidth=1.2)
                ax.axhline(num_neurons + 4.5 - num_coll, color=[0.7, 0.7, 0.7], linestyle=':')
                ax.axhline(num_neurons + 4.5 - (num_coll + num_indi), color=[0.7, 0.7, 0.7], linestyle=':')
                
                ax.set_xlim(t1, t2)
                ax.set_ylim(0.5, num_neurons + 5.5)
                
                # Ticks and labels (robustes si num_coll=0 ou num_indi=0)
                y_ticks = [1, 2, 3, 4, 5, num_neurons+4-num_coll, num_neurons+4]
                y_labels = ['Total', '$\Sigma$ NC', '$\Sigma$ Indi', '$\Sigma$ Coll', str(num_neurons), str(num_coll+1), '1']
                
                y_ticks, unique_idx = np.unique(y_ticks, return_index=True)
                y_labels = [y_labels[idx] for idx in unique_idx]
                
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=7)
                ax.set_title(f'Trial : S{st_select+1}-R{rp_select+1}', fontsize=8, fontweight='bold', pad=2)
                idx_subplot += 1
                
        plt.xlabel('Time (au)', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return CellMatrix