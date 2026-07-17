# -*- coding: utf-8 -*-
"""
Created on Mon July 13 2026
@author: Laure WOLFF
Main script to test PySpike integration on the SP Hypothesis dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from pyspike import *
from generation_dataset_sp import generate_and_plot_raster
from compute_plot_distance_matrix_performance import *
from brute_force_algorithm import *
from f_bottom_up import *
from f_simulated_annealing import *

plt.close('all')


# =========================================================================
# 1. Parameters
# =========================================================================
num_stimuli = 4        # S (Number of stimuli)
num_repetitions = 5    # R (Repetitions per stimulus)
num_neurons = 7       # N (Total number of neurons)
num_coll = 3          # Collectively coding neurons
num_indi = 0           # Individually coding neurons
t1, t2 = 0.0, 1.0      # Time window (seconds)
refrac = 0.002         # Refractory period of 2 ms
base_rate = 20         # Baseline firing rate (Hz)


num_coding_neurons = num_indi + num_coll
showing = True
plotting = True
other_figs = True

np.random.seed(12)

# =========================================================================
# 2. Creation of the dataset (Figure 1)
# =========================================================================
print("--- Generation of the dataset (SP Hypothesis) ---")
CellMatrix = generate_and_plot_raster(
    num_stimuli, num_repetitions, num_indi, num_coll, 
    num_neurons, t1, t2, base_rate, refrac, plotting, other_figs
)

# =========================================================================
# 3. Compute and plot the distances matrices and performance (Figure 2)
# =========================================================================
print ("--- The three matrices and the performance values (SP Hypothesis) ---")
plot_and_compute_distance_matrix(CellMatrix, num_neurons, num_coding_neurons, 
                                     num_stimuli, num_repetitions, t1, t2)


# =========================================================================
# 5. Individuals performance plotting (FIGURE 7C)
# Useful to check the bottom-up algorithm
# =========================================================================
print(f"--- Calcul of the individual performance of each neurons  ---")
P_individuelles = np.zeros(num_neurons)

for nc in range(num_neurons):
    sub_matrix = [cell_matrix[nc] for cell_matrix in [CellMatrix]]
    P_solo, _ = compute_distance_matrix_performance(sub_matrix, 1, num_stimuli, num_repetitions, t1, t2)
    P_individuelles[nc] = P_solo

# Plotting of the Figure 7C from the paper 2018
fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor='w')
neurons_idx = np.arange(1, num_neurons + 1)
ax3.bar(neurons_idx, P_individuelles, color=[0.30, 0.75, 0.93], edgecolor='black')

# Line to separate each groups (Coll | Indi | NC)
ax3.axvline(num_coll + 0.5, color='black', linewidth=1.5)
ax3.axvline(num_coll + num_indi + 0.5, color='black', linewidth=1.5)

ax3.set_xlim(0.5, num_neurons + 0.5)
ax3.set_ylim(0, np.max(P_individuelles) * 1.2 if np.max(P_individuelles) > 0 else 1.0)
ax3.set_xticks(range(1, num_neurons + 1))
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

ax3.set_xlabel('Neuron Index', fontsize=11, fontweight='bold')
ax3.set_ylabel('Individual Performance', fontsize=11, fontweight='bold')
ax3.set_title('Individual Performance Profile (Fig 7C)', fontsize=12, fontweight='bold')

max_y = np.max(P_individuelles) if np.max(P_individuelles) > 0 else 0.8
ax3.text(num_coll / 2 + 0.5, max_y * 1.1, 'Coll', ha='center', fontweight='bold')
ax3.text(num_coll + num_indi / 2 + 0.5, max_y * 1.1, 'Indi', ha='center', fontweight='bold')
ax3.text(num_coll + num_indi + (num_neurons - num_coll - num_indi) / 2 + 0.5, max_y * 1.1, 'NC', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# =========================================================================
# 4. Algorithms
# =========================================================================

# 4.1 Brute Force algorithm
if num_neurons < 20 :
    best_subpop, best_perf_overall = f_brute_force(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, other_figs)

# 4.2 Bottom-Up algorithm
best_subpop_print, max_P = f_bottom_up(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, plotting, other_figs)

# 4.3 Simulated annealing algorithm
nbr_iteration = f_simulated_annealing(CellMatrix, num_neurons, num_stimuli, num_repetitions, t1, t2, showing, plotting, other_figs)

# ======================================================================================================================================

# =========================================================================
# 6. Automatic Results Export
# =========================================================================
import os
from datetime import datetime

print("\n--- Saving results automatically ---")

# 1. Création d'un dossier de sauvegarde unique
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"Resultats_SP_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 2. Sauvegarde automatique de TOUTES les figures ouvertes
fig_nums = plt.get_fignums()
for num in fig_nums:
    fig = plt.figure(num)
    # On récupère le titre de la fenêtre pour donner un nom propre au fichier
    title = fig.canvas.manager.get_window_title()
    clean_title = title.replace(" ", "_").replace("-", "_").lower() if title else f"figure_{num}"
    
    fig.savefig(os.path.join(output_dir, f"{clean_title}.jpg"), dpi=300, bbox_inches='tight')

print(f"✓ {len(fig_nums)} figures saved in '{output_dir}/'")

# 3. Écriture d'un rapport textuel des performances
report_path = os.path.join(output_dir, "summary_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("      SIMULATION REPORT - SP HYPOTHESIS           \n")
    f.write(f"      Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("==================================================\n\n")
    
    f.write("--- Simulation Parameters ---\n")
    f.write(f"Neurons (N): {num_neurons} (Coll: {num_coll}, Indi: {num_indi})\n")
    f.write(f"Stimuli (S): {num_stimuli} | Repetitions (R): {num_repetitions}\n")
    f.write(f"Time window: [{t1}, {t2}] s | Base rate: {base_rate} Hz\n\n")
    
    f.write("--- Algorithms Results ---\n")
    if num_neurons < 20 :
        f.write(f"1. Brute Force :\n   Best Subpop: {list(best_subpop)}\n   Max Perf P : {best_perf_overall:.4f}\n\n")
    f.write(f"2. Bottom-Up :\n   Best Subpop: {list(best_subpop_print)}\n   Max Perf P : {max_P:.4f}\n\n")
    # Si ton SA retourne la sous-population, tu pourras l'ajouter ici de la même manière
    f.write(f"3. Simulated Annealing :\n   Iterations : {nbr_iteration}\n")

print(f"✓ Summary report saved as '{report_path}'")