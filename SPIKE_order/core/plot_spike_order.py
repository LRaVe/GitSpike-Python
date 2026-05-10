"""Plot the spike-order signal and its overall value D."""

import numpy as np
import matplotlib.pyplot as plt
from .order_spikes import order_spikes


def plot_spike_order(spikes, tmin, tmax, threshold=1e-10):
    orders = order_spikes(tmin, tmax, spikes)
    time = np.concatenate([np.asarray(s) for s in spikes])
    value = np.concatenate([np.asarray(o) for o in orders])

    orderInd = np.argsort(time)
    sortedTimes = time[orderInd]
    sortedOrders = value[orderInd]

    D = np.sum(sortedOrders)
    if abs(D) < threshold:
        D = 0.0

    plt.figure()
    plt.grid(True)
    plt.plot([tmin, tmax], [D, D], '-', color='red', linewidth=1)
    plt.plot(sortedTimes, sortedOrders, '-o', color='blue', linewidth=1.5, markersize=6)
    plt.xlim([tmin, tmax])
    plt.ylim([-1.1, 1.1])
    plt.title(f'Spike order D = {D:.4g}')
    plt.yticks([-1, 0, 1])
    plt.show()
