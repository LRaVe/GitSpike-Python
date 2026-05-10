"""Plot the spike-train-order signal and the pairwise-order matrix."""

import numpy as np
import matplotlib.pyplot as plt


def compute_spike_train_order_value(spikes, orders,number_spikes):
    """Compute the global spike-train-order value from already computed orders."""
    time = np.concatenate([np.asarray(s) for s in spikes])
    value = np.concatenate([np.asarray(o) for o in orders])
    order_ind = np.argsort(time)
    sorted_orders = value[order_ind]
    F = np.sum(sorted_orders)
    return F / number_spikes if number_spikes != 0 else F


def plot_spike_train_order(spikes, orders, order_matrix, F, tmin, tmax):
    """Plot the global spike-train-order signal and the pairwise matrix."""

    time = np.concatenate([np.asarray(s) for s in spikes])
    value = np.concatenate([np.asarray(o) for o in orders])

    orderInd = np.argsort(time)
    sortedTimes = time[orderInd]
    sortedOrders = value[orderInd]

    # ===================================================
    # ====== Plotting the spike-train-order signal ======
    # ===================================================

    plt.figure()
    plt.grid(True)
    plt.plot([tmin, tmax], [F, F], '-', color='red', linewidth=1)
    plt.plot(sortedTimes, sortedOrders, '-o', color='blue', linewidth=1.5, markersize=6)
    plt.xlim([tmin, tmax])
    plt.ylim([-1.1, 1.1])
    plt.title(f'Spike train order F = {F:.4g}')
    plt.yticks([-1, 0, 1])
    plt.show()

    # ======================================================
    # ====== Plotting the pairwise train order matrix ======
    # ======================================================

    plt.figure()
    matrix_min = np.min(order_matrix)
    matrix_max = np.max(order_matrix)
    plt.imshow(order_matrix, vmin=matrix_min, vmax=matrix_max, cmap='jet', origin='upper')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.arange(order_matrix.shape[0]), np.arange(1, order_matrix.shape[0] + 1))
    plt.yticks(np.arange(order_matrix.shape[0]), np.arange(1, order_matrix.shape[0] + 1))
    plt.xlabel('Train index')
    plt.ylabel('Train index')
    plt.title(f'Pairwise train order matrix F = {F:.4g}')
    plt.show()
