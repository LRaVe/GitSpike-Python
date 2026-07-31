"""
SP_example.py

Runnable example reproducing the Summed Population (SP) part of the
original MATLAB Main.m, using the translated PySpike algorithm
modules: dataset generation, Fig. 1/2, single-neuron performance,
brute force, bottom-up, top-down, and simulated annealing.

Based on the original MATLAB Main.m (Maxime Beltoise).
"""

import numpy as np
import matplotlib.pyplot as plt

from discriminative_neuronal_subpopulation.datasets import generate_SP_dataset
from discriminative_neuronal_subpopulation.plotting import (
    plot_SP_figure, plot_distance_matrix, plot_brute_force,
    plot_bottom_up, plot_top_down, plot_simulated_annealing,
)
from discriminative_neuronal_subpopulation.discrimination import (
    build_trials, compute_population_distance_matrix, evaluate_population,
)
from discriminative_neuronal_subpopulation.search import (
    brute_force_search, bottom_up_search, top_down_search,
)
from discriminative_neuronal_subpopulation.simulated_annealing import simulated_annealing


def run_sp_example():

    # =====================================================
    # SEED
    # =====================================================
    ind_seed = 493480      # to keep track of a seed's index

    change_seed = True     # set to False to reuse ind_seed exactly
    if change_seed:
        ind_seed = np.random.randint(1, 1_000_001)  # matches MATLAB's randi([1 1000000])

    np.random.seed(ind_seed)
    print(f'Seed used: {ind_seed}')

    # =====================================================
    # PARAMETERS
    # =====================================================

    params = {
        'N': 7, 'c': 3, 'nIndi': 0, 'indiJitter': 0.01,
        'S': 4, 'R': 5, 'Tmax': 1.0, 'rate': 10,
    }

    N, c, Tmax = params['N'], params['c'], params['Tmax']
    coding_neurons = range(0, c)
    non_coding = range(c, N)   # NB: also includes the Indi neurons
    all_neurons = range(0, N)

    # =====================================================
    # GENERATE DATASET
    # =====================================================
    
    # spikes = generate_SP_dataset(params)

    from discriminative_neuronal_subpopulation.matlab_io import load_spikes_from_mat
    spikes = load_spikes_from_mat("SP_python_data.mat", spikes_var = "SP_python_data",Tmax= Tmax)

    # =====================================================
    # FIGURE 1
    # =====================================================
    plot_params = {
            'stimuli': [0, 1], 'repetitions': [0, 1], 'showPooling': True,
        }
    
    plot_SP_figure(spikes, params, plot_params)

    # =====================================================
    # FIGURE 2
    # =====================================================
    trials_c, labels = build_trials(spikes, coding_neurons)
    D_c = compute_population_distance_matrix(trials_c, Tmax)
    trials_nc, _ = build_trials(spikes, non_coding)
    D_nc = compute_population_distance_matrix(trials_nc, Tmax)
    trials_all, _ = build_trials(spikes, all_neurons)
    D_all = compute_population_distance_matrix(trials_all, Tmax)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    plot_distance_matrix(D_c, labels, 'C', ax=axes2[0])
    plot_distance_matrix(D_nc, labels, 'NC', ax=axes2[1])
    plot_distance_matrix(D_all, labels, 'All', ax=axes2[2])
    fig2.tight_layout()

    # =====================================================
    # SINGLE NEURON PERFORMANCE
    # =====================================================
    P_single = np.zeros(N)
    for n in range(N):
        P_single[n], _, _ = evaluate_population(spikes, [n], Tmax)

    print('Neuron numbers :', N)
    print('Psingle :', np.round(P_single, 3))

    fig10, ax10 = plt.subplots()
    ax10.bar(np.arange(1, N + 1), P_single)
    ax10.set_xlabel('Neuron')
    ax10.set_ylabel('P')
    ax10.set_title('Single neuron performance')
    ax10.grid(True)

    # =====================================================
    # BRUTE FORCE
    # =====================================================
    if N < 14:
        print('Starting Brute Force...')
        bf_result = brute_force_search(spikes, Tmax, verbose=True)
        plot_brute_force(bf_result)
    else:
        print(f'Skipping Brute Force (N={N} too large).')

    # =====================================================
    # BOTTOM-UP
    # =====================================================
    print('Starting Bottom-Up...')
    bu_result = bottom_up_search(spikes, Tmax, verbose=True)
    plot_bottom_up(bu_result, other_figs=False)

    # =====================================================
    # TOP-DOWN
    # =====================================================
    print('Starting Top-Down...')
    td_result = top_down_search(spikes, Tmax)
    print('Best population found :', td_result['bestPopulation'])
    print(f"Best P = {td_result['bestP']:.4f}")
    plot_top_down(td_result, coding_neurons)

    # =====================================================
    # SIMULATED ANNEALING
    # =====================================================
    params_sa = {
        'N0': 200, 'steps': 10 * N, 'coolingFactor': 0.95,
        'codingNeurons': list(coding_neurons),
    }
    print('Starting Simulated Annealing...')
    sa_result = simulated_annealing(spikes, Tmax, params_sa, verbose=True)
    print('SA best population :', sa_result['bestPopulation'])
    print(f"SA best P = {sa_result['bestP']:.4f}")
    plot_simulated_annealing(sa_result, coding_neurons)


if __name__ == '__main__':
    run_sp_example()
    plt.show()