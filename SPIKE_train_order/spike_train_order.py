"""Example entry point for the spike-train-order computation.

This module builds a small demo dataset, adds auxiliary spikes, computes the
train order, and displays the resulting plots.
"""

from spike_common import add_auxiliary_spikes
from .core.order_trains import order_trains
from .core.plot_spike_train_order import compute_spike_train_order_value, plot_spike_train_order


def spike_train_order():
    tmin = 0
    tmax = 10
    spikes = []
    spikes.append([0, 1.9, 3.9, 7, 10])
    spikes.append([0, 2, 7.1, 9, 10])
    spikes.append([0, 2.1, 4.1, 6.9, 10])
    spikes.append([0, 2.2, 6.8, 7.1, 10])
    #spikes.append([0.0001, 0.7142])
    #spikes.append([0.2858, 0.9999])
    #spikes.append([0.1429, 0.8571])

    number_spikes=sum(len(s) for s in spikes)

    spikes, _, _ = add_auxiliary_spikes(spikes, tmin, tmax)

    results, order_matrix = order_trains(tmin, tmax, spikes)

    F = compute_spike_train_order_value(spikes, results,number_spikes)

    try:
        plot_spike_train_order(spikes, results, order_matrix, F, tmin, tmax)
    except Exception:
        pass

    print("Spike-train-order results:\n", [result.tolist() for result in results])
    print("Pairwise train order matrix:\n", order_matrix)
    print("Spike-train-order F=", f"{F:.4g}")

    return results, order_matrix

if __name__ == '__main__':
    spike_train_order()
    
