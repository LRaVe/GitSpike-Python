"""
LL_example.py

Runnable example reproducing the Labeled Line (LL) part of the
original MATLAB Main.m (Fig. 9 configuration): dataset generation,
evaluate_LL_population, plot_LL_results.

Based on the original MATLAB Main.m (Maxime Beltoise).
"""

import numpy as np
import matplotlib.pyplot as plt

from pyspike.algo.datasets import generate_LL_dataset
from pyspike.algo.labeled_line import evaluate_LL_population
from pyspike.algo.plotting import plot_LL_results


def run_ll_example():

    # =====================================================
    # SEED
    # =====================================================
    ind_seed = 493480      # to keep track of a seed's index

    change_seed = True     # set to False to reuse ind_seed exactly
    if change_seed:
        ind_seed = np.random.randint(1, 1_000_001)  # matches MATLAB's randi([1 1000000])

    np.random.seed(ind_seed)
    print(f'Seed used: {ind_seed}')

    
    # Fig. 9 configuration from Main.m
    params_ll = {
        'mode': 'structured',
        'N': 10, 'S': 8, 'R': 5,
        'Tmax': 1.0,
        'meanRate': 20,
        'jitter': 0.01,
        'jitterIntensity': [1, 1/3, 1.5, 1/3, 2, 2, 2, 1/3, 1/2, 1.5],
        'responseMatrix': np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ]),
        'sameResponse': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    }

    # =====================================================
    # GENERATE DATASET
    # =====================================================
    spikes, response_matrix = generate_LL_dataset(params_ll)

    #from pyspike.algo.matlab_io import load_spikes_from_mat
    #spikes = load_spikes_from_mat('Dataset_Fig4.mat')


    # =====================================================
    # EVALUATE LL POPULATION
    # =====================================================
    result_ll = evaluate_LL_population(spikes, params_ll['Tmax'])

    print('Best population found :', [n + 1 for n in result_ll['bestPopulation']])
    print(f"Best P = {result_ll['bestP']:.4f}")

    # =====================================================
    # DISPLAY
    # =====================================================
    plot_LL_results(spikes, result_ll)


if __name__ == '__main__':
    run_ll_example()
    plt.show()