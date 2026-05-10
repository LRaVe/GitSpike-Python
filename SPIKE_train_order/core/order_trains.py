import numpy as np
from .pairwise_train_order import pairwise_train_order


def order_trains(tmin, tmax, spikes):
    n = len(spikes)
    results = [None] * n
    order_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        results[i] = np.zeros(len(spikes[i]), dtype=float)
        order_matrix[i, i] = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            res_i, res_j = pairwise_train_order(tmin, tmax, spikes, i + 1, j + 1)
            results[i] = results[i] + res_i
            results[j] = results[j] + res_j
            order_matrix[i, j] = np.mean(res_i) if res_i.size > 0 else 0.0
            order_matrix[j, i] = -order_matrix[i, j]

    if n > 1:
        for i in range(n):
            results[i] = results[i] / (n - 1)

    return results, order_matrix
