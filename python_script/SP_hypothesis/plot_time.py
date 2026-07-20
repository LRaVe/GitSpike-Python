# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 2026
@author: Laure WOLFF
Benchmark Script for Figure 8: Search Space Exploration Scale
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from generation_dataset_sp import generate_and_plot_raster
from compute_plot_distance_matrix_performance import compute_distance_matrix_performance
import compute_plot_distance_matrix_performance as distance_module
from brute_force_algorithm import f_brute_force
from f_bottom_up import f_bottom_up
from f_simulated_annealing import f_simulated_annealing

plt.close('all')

# =========================================================================
# 1. PARAMÈTRES GÉNÉRAUX & BENCHMARK
# =========================================================================
num_stimuli = 4        
num_repetitions = 5    
t1, t2 = 0.0, 1.0      
refrac = 0.002         
base_rate = 20         
np.random.seed(42)

# Échelle de tailles de pools à tester (Axe X)
neuron_counts = [3, 5, 8, 10, 12, 15, 20, 25, 30 ]

# Tableaux de stockage des courbes
eval_brute = np.zeros(len(neuron_counts))
eval_greedy = np.zeros(len(neuron_counts))
eval_sa_total = np.zeros(len(neuron_counts))
eval_sa_unique = np.zeros(len(neuron_counts))

print("=====================================================================")
print("STARTING BENCHMARK FOR FIGURE 8 (EXPLORATION COMPLEXITY)")
print("=====================================================================")

# =========================================================================
# 2. BOUCLE DE BENCHMARKING
# =========================================================================
for idx, N in enumerate(neuron_counts):
    print(f"\n---> Testing Population Size: N = {N} Neurons")
    
    # Génération du dataset SP spécifique à cette taille de pool N
    num_coll = max(1, N // 2)
    num_indi = 0
    CellMatrix = generate_and_plot_raster(
        num_stimuli, num_repetitions, num_indi, num_coll, 
        N, t1, t2, base_rate, refrac, plotting=False, other_figs=False
    )
    
    # -----------------------------------------------------------------
    # 2.1 COMPLEXITÉ ALGORITHME 1 : BRUTE FORCE
    # -----------------------------------------------------------------
    eval_brute[idx] = (2 ** N) - 1
    print(f"  [Brute Force]        Evaluations = {int(eval_brute[idx])}")
    
    # -----------------------------------------------------------------
    # 2.2 COMPLEXITÉ ALGORITHME 2 : GRADIENT (BOTTOM-UP)
    # -----------------------------------------------------------------
    eval_greedy[idx] = (N * (N + 1)) / 2
    print(f"  [Bottom-Up]          Evaluations = {int(eval_greedy[idx])}")
    
    # -----------------------------------------------------------------
    # 2.3 COMPLEXITÉ ALGORITHME 3 : RECUT SIMULÉ (TOTAL VS UNIQUE)
    # -----------------------------------------------------------------
    import f_simulated_annealing as sa_module 
    
    original_compute_func = distance_module.compute_distance_matrix_performance
    
    sa_runs_total = []
    sa_runs_unique = []
    
    # Moyenne sur 3 lancements pour lisser la stochasticité
    for run in range(3):
        tracker = {
            'total_eval': 0,
            'unique_masks': set()
        }
        
        def wrapped_compute_distance(cell_matrix, selection_mask, *args, **kwargs):
            mask_tuple = tuple(np.array(selection_mask).astype(int))
            if np.sum(mask_tuple) > 0:  
                tracker['total_eval'] += 1
                tracker['unique_masks'].add(mask_tuple)
            return original_compute_func(cell_matrix, selection_mask, *args, **kwargs)
        
        # Double injection pour intercepter les calculs internes du SA
        distance_module.compute_distance_matrix_performance = wrapped_compute_distance
        sa_module.compute_distance_matrix_performance = wrapped_compute_distance
        
        # Exécution silencieuse
        _ = f_simulated_annealing(CellMatrix, N, num_stimuli, num_repetitions, t1, t2, 
                                  showing=False, plotting=False, other_figs=False)
        
        sa_runs_total.append(tracker['total_eval'])
        sa_runs_unique.append(len(tracker['unique_masks']))
        
    # Restauration propre des fonctions originales
    distance_module.compute_distance_matrix_performance = original_compute_func
    sa_module.compute_distance_matrix_performance = original_compute_func
    
    eval_sa_total[idx] = np.mean(sa_runs_total)
    eval_sa_unique[idx] = np.mean(sa_runs_unique)
    
    print(f"  [Simulated Annealing] Total Visited = {eval_sa_total[idx]:.1f} | Unique = {eval_sa_unique[idx]:.1f}")

print("\nBenchmark successfully completed!")

# =========================================================================
# 3. GRAPHISME CONFORME À LA PUBLICATION (FIGURE 8)
# =========================================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor='w')
fig.canvas.manager.set_window_title('Figure 8 - Algorithmic Search Space Exploration Scale')

# 1. Courbe Gradient / Bottom-Up (Cercles bleus d'origine - Éq. 7)
ax.plot(neuron_counts, eval_greedy, '-o', color='#004474', linewidth=2, markersize=7,
        markerfacecolor='#004474', label='Gradient algorithms (Two variants, Eq. (7))')

# 2. Courbe Simulated Annealing - Marche réelle (Carrés orange - Visités)
ax.plot(neuron_counts, eval_sa_total, '-s', color='#D95319', linewidth=2, markersize=7,
        markerfacecolor='#D95319', label='Simulated annealing (Actual number visited)')

# 3. Courbe Simulated Annealing - Uniques (Triangles rouges - Corrigé !)
ax.plot(neuron_counts, eval_sa_unique, '-^', color='r', linewidth=2, markersize=7,
        markerfacecolor='r', label='Simulated annealing (Uniquely evaluated)')

# 4. Courbe Brute Force (Losanges Violets)
ax.plot(neuron_counts, eval_brute, '-d', color='#7E2F8E', linewidth=2, markersize=7,
        markerfacecolor='#7E2F8E', label='Brute force')

# Configuration de l'échelle logarithmique et de la grille
ax.set_yscale('log')
ax.set_xticks(neuron_counts)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Titres et labels du texte d'origine
ax.set_xlabel('Number of neurons ($N$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of evaluated subpopulations', fontsize=12, fontweight='bold')
ax.set_title('Figure 8: Evaluation count comparison between different search algorithms', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=False)

plt.tight_layout()

# =========================================================================
# 4. EXPORTATION AUTOMATIQUE DES RÉSULTATS
# =========================================================================
print("\n--- Saving Figure 8 automatically ---")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"Resultats_Figure8_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

fig.savefig(os.path.join(output_dir, "figure_8_complexity.jpg"), dpi=300, bbox_inches='tight')
print(f"✓ Figure 8 saved in '{output_dir}/figure_8_complexity.jpg'")

plt.show()