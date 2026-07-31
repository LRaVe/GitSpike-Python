"""
benchmark_example.py

Runnable example reproducing the original MATLAB time_plot_V2.m:
compares the number of subpopulations evaluated by the SP search
algorithms as a function of the number of neurons N (Fig. 8).

Please note this can take a while to run, especially for larger
neuron_counts, since Simulated Annealing is actually executed (unlike
the Bottom-Up/Top-Down and Brute Force curves, which are theoretical
closed-form values — see run_complexity_benchmark's docstring).

Based on the original MATLAB time_plot_V2.m (Laure Wolff).
"""

import matplotlib.pyplot as plt

from search import run_complexity_benchmark
from plotting import plot_complexity_benchmark


def run_benchmark_example():

    neuron_counts = [5, 10, 15, 20, 25]

    sp_params = {
        'S': 4, 'R': 5, 'Tmax': 1.0, 'rate': 10,
    }

    sa_params = {
        'N0': 50, 'coolingFactor': 0.9,
    }

    result = run_complexity_benchmark(
        neuron_counts, sp_params, sa_params, loop=10, steps_per_neuron=5,
        verbose=True)

    plot_complexity_benchmark(result)

    return result


if __name__ == '__main__':
    run_benchmark_example()
    plt.show()