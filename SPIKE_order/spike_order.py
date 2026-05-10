"""Example entry point for the spike-order computation.

This module builds a small demo dataset, adds auxiliary spikes, computes the
spike order, and displays the resulting plot.
"""

from spike_common import add_auxiliary_spikes
from .core.order_spikes import order_spikes
from .core.plot_spike_order import plot_spike_order

def spike_order():
    threshold = 1e-10
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

    spikes, _, _ = add_auxiliary_spikes(spikes, tmin, tmax)
    results = order_spikes(tmin, tmax, spikes)

    try:
        # If called as script, show plot
        plot_spike_order(spikes, tmin, tmax, threshold)
    except Exception:
        pass

    print("Spike-order results:\n", [result.tolist() for result in results])
    
    return results

if __name__ == '__main__':
    spike_order()
