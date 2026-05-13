# -*- coding: utf-8 -*-
"""
ISI-distance adaptive computation with auxiliary boundary spikes and plotting 

Created on Wed May 13 09:21:35 2026

@author: Laure WOLFF
"""

import numpy as np
import matplotlib.pyplot as plt
import pyspike as spk

def calculate_auto_mrts(spikes_trains):
    """Calculate the mean of the smallest ISI-distance of the dataset"""
    min_isis = []
    for train in spikes_trains:
        if len(train) > 1:
            min_isis.append(np.min(np.diff(train)))
    
    return np.mean(min_isis) if min_isis else 0.0

def f_isi_distance_adaptive(spikes_trains, tmin, tmax, MRTS=0):
    # Manages the MRTS parameter
    if isinstance(MRTS, str) and MRTS.lower() == 'auto':
        MRTS = calculate_auto_mrts(spikes_trains)
        mode_label = f"Adaptive (auto MRTS = {MRTS:.3f})"
    elif MRTS > 0:
        mode_label = f"Adaptive (manual MRTS = {MRTS:.3f})"
    else:
        mode_label = "Classic (MRTS = 0)"

    num_trains = len(spikes_trains)
    dist_matrix = np.zeros((num_trains, num_trains))
    all_t_events = [tmin, tmax]
    pair_data = []

    # Edge correction
    spikes = []
    for train in spikes_trains:
        s = np.unique(train)
        spikes.append(s[(s > tmin) & (s < tmax)])

    if num_trains >= 2:
        num_pairs = int(num_trains * (num_trains - 1) / 2)
        num_cols = 2
        num_rows = int(np.ceil(num_pairs / num_cols))
        
        fig_pairs, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows), squeeze=False)
        fig_pairs.suptitle(f"ISI Evolution - {mode_label}")
        
        pair_idx = 0
        distances_list = []

        for i in range(num_trains):
            for j in range(i + 1, num_trains):
                t_all = np.unique(np.concatenate(([tmin], spikes[i], spikes[j], [tmax])))
                all_t_events.extend(t_all)
                
                iij_sum = 0
                it_list = []

                for k in range(len(t_all) - 1):
                    t_mid = (t_all[k] + t_all[k+1]) / 2
                    
                    # Train i
                    if len(spikes[i]) == 0 or t_mid < spikes[i][0]:
                        val_x = max(spikes[i][0] - tmin if len(spikes[i]) > 0 else 0, MRTS)
                    elif t_mid > spikes[i][-1]:
                        val_x = max(tmax - spikes[i][-1], MRTS)
                    else:
                        idx = np.where(spikes[i] <= t_mid)[0][-1]
                        val_x = max(spikes[i][idx+1] - spikes[i][idx], MRTS)
                    
                    # Train j
                    if len(spikes[j]) == 0 or t_mid < spikes[j][0]:
                        val_y = max(spikes[j][0] - tmin if len(spikes[j]) > 0 else 0, MRTS)
                    elif t_mid > spikes[j][-1]:
                        val_y = max(tmax - spikes[j][-1], MRTS)
                    else:
                        idy = np.where(spikes[j] <= t_mid)[0][-1]
                        val_y = max(spikes[j][idy+1] - spikes[j][idy], MRTS)

                    i_t = abs(val_x - val_y) / max(val_x, val_y)
                    iij_sum += i_t * (t_all[k+1] - t_all[k])
                    it_list.append(i_t)

                dist_val = iij_sum / (tmax - tmin)
                dist_matrix[i, j] = dist_val
                dist_matrix[j, i] = dist_val
                distances_list.append(dist_val)
                
                pair_data.append({'t': t_all, 'It': it_list})

                # Plotting pairwise ISI-distance
                ax = axes[pair_idx // num_cols, pair_idx % num_cols]
                ax.step(t_all, it_list + [it_list[-1]], where='post', lw=1.5)
                ax.set_title(f"Pair {i} & {j}\nDist: {dist_val:.3f}")
                ax.set_xlim(0, tmax)
                ax.set_ylim(0, 1)
                ax.grid(True)
                pair_idx += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Average calculation
        t_global = np.unique(all_t_events)
        i_matrix = np.zeros((len(pair_data), len(t_global) - 1))
        
        for p, data in enumerate(pair_data):
            t_p = data['t']
            it_p = data['It']
            for k in range(len(t_global) - 1):
                t_mid = (t_global[k] + t_global[k+1]) / 2
                idx = np.where(t_p[:-1] <= t_mid)[0][-1]
                i_matrix[p, k] = it_p[idx]
        
        i_pop_mean = np.mean(i_matrix, axis=0)

        # Plot Matrix
        plt.figure()
        plt.imshow(dist_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f"ISI-distance Matrix - {mode_label}")

        # Plot Pop Average
        plt.figure()
        plt.step(t_global, np.append(i_pop_mean, i_pop_mean[-1]), where='post')
        plt.xlabel('Time')
        plt.ylabel('Average I_t')
        plt.suptitle(f"Population Average ISI distance - {mode_label}")
        plt.title(f"Average ISI-distance: {np.mean(distances_list):.3f}")
        plt.xlim(0, tmax)
        plt.ylim(0, 1)
        plt.show()

        return dist_matrix, distances_list, np.mean(distances_list)
    else:
        return dist_matrix, [], 0.0

# test
if __name__ == "__main__":
    tmin, tmax = 0, 10
    num_trains = 3
    spikes = [
        [0, 1, 2, 4, 7],
        [3, 4, 6, 10],
        [2, 5]
    ]
    f_isi_distance_adaptive(spikes, 0, 10, MRTS='auto')
    f_isi_distance_adaptive(spikes, 0, 10, MRTS=0)
    f_isi_distance_adaptive(spikes, 0, 10, MRTS=1.5)