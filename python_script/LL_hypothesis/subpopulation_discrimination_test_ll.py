# -*- coding: utf-8 -*-
"""
Created on Tue July 14 2026
@author: Laure WOLFF
Main script to test PySpike integration on the LL (Labeled Line) Hypothesis dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from time import *
from pyspike import *
from genration_dataset_ll import *
from pairwise_matices_D import *
from compute_plot_matrices_M import *
from compute_plot_performance_matrices import *


plt.close('all')

# =========================================================================
# 1. Parameters
# =========================================================================
num_stimuli = 4        # S (Number of stimuli)
num_repetitions = 5    # R (Repetitions per stimulus)
num_neurons = 4        # N (Total number of neurons)
num_indi = 4           # Individually coding neurons (LL-specific)
t1, t2 = 0.0, 1.0      # Time window (seconds)
refrac = 0.002         # Refractory period (2 ms)
base_rate = 20         # Baseline firing rate (Hz)

# Specific parameter for the LL model (temporal jitter of spikes)
jitter_std = 0.005     # Jitter standard deviation (5 ms)

showing = True
plotting = True
other_figs = True       # Allows opening the individual raster plot figure

np.random.seed(12)

# =========================================================================
# 2. Creation of the dataset 
# =========================================================================
# print("--- Generation of the dataset (LL Hypothesis) ---")
# CellMatrix = generate_and_plot_raster_ll(
#     num_stimuli=num_stimuli,
#     num_repetitions=num_repetitions,
#     num_indi=num_indi,
#     num_neurons=num_neurons,
#     t1=t1,
#     t2=t2,
#     base_rate=base_rate,
#     refrac=refrac,
#     jitter_std=jitter_std,
#     showing=showing,
#     plotting=plotting,
#     other_figs=other_figs
# )

# # Used for debugging MATLAB code 
# import scipy.io

# CellMatrix_to_export = np.empty(CellMatrix.shape, dtype=object)
# for n in range(num_neurons):
#     for st in range(num_stimuli):
#         for rp in range(num_repetitions):
#             CellMatrix_to_export[n, st, rp] = np.array(CellMatrix[n, st, rp], dtype=np.float64).flatten()

# # Save to the MATLAB's format
# scipy.io.savemat('LL_python_data.mat', {'LL_python_data': CellMatrix_to_export}, oned_as='row')
# print("✓ LL_python_data.mat successful exported for MATLAB !")

# # Other dataset

# from scipy import io
# mat_data = io.loadmat("LL.mat")
# CellMatrix = mat_data["CELL"]

# # print("Shape de CellMatrix :", CellMatrix.shape)
# # print("Type du premier essai :", type(CellMatrix[0, 0, 0]))
# # print("Contenu réel du premier essai [0, 0, 0] :", CellMatrix[0, 0, 0])

import scipy.io

# Load the MATLAB file
mat_data = scipy.io.loadmat('LL_matlab_data.mat')
raw_cell = mat_data['CellMatrix']

num_neurons, num_stimuli, num_repetitions = raw_cell.shape

# Reconstruct CellMatrix with SpikeTrain objects
CellMatrix = np.empty((num_neurons, num_stimuli, num_repetitions), dtype=object)

for n in range(num_neurons):
    for s in range(num_stimuli):
        for r in range(num_repetitions):
            # Extract times from the MATLAB cell
            times = raw_cell[n, s, r].flatten() if raw_cell[n, s, r].size > 0 else np.array([], dtype=np.float64)
            
            # Wrap into SpikeTrain
            CellMatrix[n, s, r] = SpikeTrain(times, [t1, t2])
        
# =========================================================================
# 3. Creation of the pairwise matrix and M matrices (Wilcoxon test)
# =========================================================================

print("\n--- Computing SPIKE-distance matrix ---")
All_Matrix_D = SPIKE_Distance_matrix(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, plotting)

# print(np.sum(All_Matrix_D))
# print(np.mean(All_Matrix_D))

# Used for debugging MATLAB code 
# import scipy.io
# scipy.io.savemat('distances_python.mat', {'All_Matrix_D_py': All_Matrix_D})

print("\n--- Computing Wilcoxon discrimination matrices (M) ---")
All_Matrices_M = calculate_plot_matrix_M(All_Matrix_D, num_neurons, num_stimuli, num_repetitions, plotting)

print("\n--- Computing performance matrices (P) & best neuron mapping ---")
P_all_neurons, P_pop, M_max, opt_LL, PLL_total = calculate_and_plot_performance_matrix(
    All_Matrices_M, All_Matrix_D, num_neurons, num_stimuli, num_repetitions, showing, plotting
)

# =========================================================================
# 4. Automatic Results Export (Raster and performance figures)
# =========================================================================
print("\n--- Saving all figures automatically ---")

# Cerate a unique folder to save each run of this program
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"Resultats_LL_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save all the figure opened during the run
fig_nums = plt.get_fignums()
for num in fig_nums:
    fig = plt.figure(num)
    title = fig.canvas.manager.get_window_title()
    clean_title = title.replace(" ", "_").replace("-", "_").replace("{", "").replace("}", "").lower() if title else f"figure_{num}"
    
    fig.savefig(os.path.join(output_dir, f"{clean_title}.jpg"), dpi=300, bbox_inches='tight')

print(f"✓ {len(fig_nums)} figure(s) saved in '{output_dir}/'")

# Writing the conclusion of each run 
report_path = os.path.join(output_dir, "generation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("     DATA GENERATION REPORT - LL HYPOTHESIS       \n")
    f.write(f"      Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("==================================================\n\n")
    
    f.write("--- Simulation Parameters ---\n")
    f.write(f"Neurons (N)     : {num_neurons} (Coding/Indi: {num_indi}, Noise: {num_neurons - num_indi})\n")
    f.write(f"Stimuli (S)     : {num_stimuli}\n")
    f.write(f"Repetitions (R) : {num_repetitions}\n")
    f.write(f"Time window     : [{t1}, {t2}] s\n")
    f.write(f"Base rate        : {base_rate} Hz\n")
    f.write(f"Jitter (std)    : {jitter_std} s\n\n")
    f.write("--- Labeled Line Results ---\n")
    f.write(f"Optimized Subpopulation (opt_LL) : {list(opt_LL)}\n")
    f.write(f"Global LL Performance (PLL_total) : {PLL_total:.4f}\n")
    f.write("\nDataset successfully processed.")

print(f"✓ Generation report saved as '{report_path}'")