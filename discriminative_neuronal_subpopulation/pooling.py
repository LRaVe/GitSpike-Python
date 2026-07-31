# Module for pooling spike trains from a subpopulation of neurons.
# Based on original MATLAB code (Maxime Beltoise), translated for PySpike.
# Distributed under the BSD License

import numpy as np


############################################################
# pool_neurons
############################################################
def pool_neurons(spikes, neurons, s, r):
    """ Pools (merges) the spike times of the given neurons for a single
    stimulus/repetition pair into one sorted array.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`, as returned by
                   :func:`.generate_SP_dataset` /
                   :func:`.generate_LL_dataset`.
    :param neurons: iterable of neuron indices (0-indexed) to pool.
    :param s: stimulus index (0-indexed).
    :param r: repetition index (0-indexed).
    :returns: sorted `np.ndarray` of the pooled spike times.
    """
    pooled = np.concatenate([spikes[n, s, r].spikes for n in neurons]) \
        if len(list(neurons)) > 0 else np.array([])

    return np.sort(pooled)