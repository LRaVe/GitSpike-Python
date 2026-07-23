# -*- coding: utf-8 -*-
"""
Created on Tue July 14 2026
@author: Laure WOLFF
Dataset generation under the Labeled Line (LL) Hypothesis 
"""

import numpy as np
import matplotlib.pyplot as plt
from pyspike import SpikeTrain

def generate_and_plot_raster_ll(num_stimuli, num_repetitions, num_indi, num_neurons, 
                               t1, t2, base_rate, refrac, jitter_std, 
                               showing=True, plotting=True, other_figs=True):
    """
    Generates a Labeled Line spike train dataset where coding neurons have specific 
    stimulus preference profiles, and plots the individual rasters.
    """
    
    if num_indi > num_neurons:
        raise ValueError("Error: The number of individual coding neurons cannot exceed total num_neurons!")
    
    num_trials = num_stimuli * num_repetitions
    
    # Initialization the 3D matrix ( Neurones x Stimuli x Repetitions)
    #CellMatrix = [[[None for _ in range(num_repetitions)] for _ in range(num_stimuli)] for _ in range(num_neurons)]
    # Remplace ton initialisation par celle-ci au début de ton générateur :
    CellMatrix = np.empty((num_neurons, num_stimuli, num_repetitions), dtype=object)
    
    # =========================================================================
    # 1. Generation of the Labeled Line Preference Matrix
    # =========================================================================
    if num_indi == 4 and num_stimuli == 4:
        pref_matrix = np.array([
            [1, 1, 0, 0],  # Neuron 1: Sensitive to S1, S2
            [0, 0, 1, 1],  # Neuron 2: Sensitive to S3, S4
            [1, 0, 1, 0],  # Neuron 3: Sensitive to S1, S3
            [0, 1, 0, 1]   # Neuron 4: Sensitive to S2, S4
        ])
    else:
        pref_matrix = (np.random.rand(num_indi, num_stimuli) < 0.5).astype(int)
        
        # Securities
        for c_idx in range(num_indi):
            if np.sum(pref_matrix[c_idx, :]) == 0:
                pref_matrix[c_idx, np.random.randint(num_stimuli)] = 1
                
        for s_idx in range(num_stimuli):
            if np.sum(pref_matrix[:, s_idx]) == 0:
                pref_matrix[np.random.randint(num_indi), s_idx] = 1
                
    if showing:
        print(f"\n--- Labeled Line Preference Matrix (Neurons {num_indi} x {num_stimuli} Stimuli) ---")
        print(pref_matrix)
        print("----------------------------------------------------------\n")
        
    # =========================================================================
    # 2. Spikes generation for CODING neurons
    # =========================================================================
    for c_idx in range(num_indi):
        # --- Creation of the baseline for each neurons---
        approx_spikes = int(round((t2 - t1) * base_rate * 2) + 10)
        uniform_samples = np.random.rand(approx_spikes)
        intervals = refrac - np.log(1 - uniform_samples) / base_rate
        baseline_spikes = t1 + np.cumsum(intervals)
        baseline_spikes = baseline_spikes[(baseline_spikes >= t1) & (baseline_spikes <= t2)]
        num_spikes = len(baseline_spikes)
        
        if num_spikes == 0:
            num_spikes = 1
            baseline_spikes = np.array([(t1 + t2) / 2.0])
            
        for st in range(num_stimuli):
            if pref_matrix[c_idx, st] == 1:
                # Jitter 
                for rp in range(num_repetitions):
                    shifts = np.random.randn(num_spikes) * jitter_std
                    jittered_train = baseline_spikes + shifts
                    
                    jittered_train = np.clip(jittered_train, t1, t2)
                    jittered_train = np.sort(jittered_train)
                    
                    CellMatrix[c_idx][st][rp] = SpikeTrain(jittered_train, [t1, t2])
            else:
                # Poisson's law 
                low_noise_rate = base_rate * 2.5
                approx_spikes_noise = int(round((t2 - t1) * low_noise_rate * 2) + 10)
                for rp in range(num_repetitions):
                    uniform_samples_noise = np.random.rand(approx_spikes_noise)
                    intervals_noise = refrac - np.log(1 - uniform_samples_noise) / low_noise_rate
                    spikes_noise = t1 + np.cumsum(intervals_noise)
                    spikes_noise = spikes_noise[(spikes_noise >= t1) & (spikes_noise <= t2)]
                    
                    CellMatrix[c_idx][st][rp] = SpikeTrain(spikes_noise, [t1, t2])
                    
    # =========================================================================
    # 3. Generation of NON-CODING background neurons
    # =========================================================================
    for c_idx in range(num_indi, num_neurons):
        for st in range(num_stimuli):
            low_noise_rate = base_rate * 2.5
            approx_spikes_noise = int(round((t2 - t1) * low_noise_rate * 2) + 10)
            for rp in range(num_repetitions):
                uniform_samples_noise = np.random.rand(approx_spikes_noise)
                intervals_noise = refrac - np.log(1 - uniform_samples_noise) / low_noise_rate
                spikes_noise = t1 + np.cumsum(intervals_noise)
                spikes_noise = spikes_noise[(spikes_noise >= t1) & (spikes_noise <= t2)]
                
                CellMatrix[c_idx][st][rp] = SpikeTrain(spikes_noise, [t1, t2])
                
    # =========================================================================
    # PLOTTING SECTION
    # =========================================================================
    if plotting:
        cols = int(np.ceil(np.sqrt(num_neurons * 1.25)))
        rows = int(np.ceil(num_neurons / cols))
        
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8), facecolor='w', sharex=True)
        fig.canvas.manager.set_window_title('Individual Neuronal Raster Plots (LL Mode)')
        axs = np.atleast_1d(axs).flatten()
        
        for n in range(num_neurons):
            ax = axs[n]
            for st in range(num_stimuli):
                y_start = st * num_repetitions + 0.5
                y_end = (st + 1) * num_repetitions + 0.5
                if st % 2 == 0:
                    ax.axhspan(y_start, y_end, color='#F5F5F5', alpha=0.5, zorder=0)
            t_counter = 1
            for st in range(num_stimuli):
                for rp in range(num_repetitions):
                    spikes = CellMatrix[n][st][rp].spikes
                    if len(spikes) > 0:
                        ax.vlines(spikes, t_counter - 0.4, t_counter + 0.4, colors='k', linewidth=0.8, zorder=2)
                    t_counter += 1
            
            # Lignes de délimitation dashed horizontales entre les stimuli
            for st_sep in range(1, num_stimuli):
                sep_line = st_sep * num_repetitions + 0.5
                ax.axhline(sep_line, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                
            ax.set_xlim(t1, t2)
            ax.set_ylim(0.5, num_trials + 0.5)
            ax.invert_yaxis()  # Pour avoir l'essai 1 en haut comme sur MATLAB
            
            # Labels Y (S1, S2...)
            if num_stimuli <= 8:
                y_ticks = np.arange(num_repetitions / 2 + 0.5, num_trials, num_repetitions)
                y_labels = [f"S{s+1}" for s in range(num_stimuli)]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=7)
            else:
                ax.set_yticks([1, num_trials])
                ax.set_yticklabels(['1', str(num_trials)], fontsize=7)
                
            # Titles
            if n < num_indi:
                ax.set_title(f"Neuron {n+1} (Coding)", color='#D62728', fontsize=9, fontweight='bold')
            else:
                ax.set_title(f"Neuron {n+1} (Noise)", color='#1F77B4', fontsize=9, fontweight='normal')
                
            # Manages the scales
            if n % cols == 0:
                ax.set_ylabel('Stimuli / Trials', fontsize=8)
            if n >= (num_neurons - cols):
                ax.set_xlabel('Time (s)', fontsize=8)
                
        for empty_ax in axs[num_neurons:]:
            empty_ax.set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    return CellMatrix