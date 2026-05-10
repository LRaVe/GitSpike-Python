"""Compute one order vector per spike train by aggregating pairwise results."""

import numpy as np
from .pairwise_order import pairwise_order


def order_spikes(tmin, tmax, spikes):
    n = len(spikes)
    results = [None] * n
    for i in range(n):
        aggregated = np.zeros(len(spikes[i]), dtype=float)
        for j in range(n):
            if i != j:
                pairwise = pairwise_order(tmin, tmax, spikes, i + 1, j + 1)
                aggregated = aggregated + pairwise
        results[i] = aggregated / (n - 1) if n > 1 else aggregated
    return results
