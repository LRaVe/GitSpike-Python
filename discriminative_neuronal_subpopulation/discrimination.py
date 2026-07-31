# Module for computing trials, distance matrices and discrimination
# performance for the discriminative subpopulation analysis.
# Based on original MATLAB code (Maxime Beltoise), translated for PySpike.
# Distributed under the BSD License

import numpy as np

from pyspike import SpikeTrain, spike_distance_matrix
from discriminative_neuronal_subpopulation.pooling import pool_neurons


############################################################
# build_trials
############################################################
def build_trials(spikes, neurons):
    """ Pools the given neurons for every stimulus/repetition pair into
    a flat list of trials, together with their stimulus labels.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param neurons: iterable of neuron indices (0-indexed) to pool.
    :returns: (trials, labels)

              - trials: list of length T=S*R, each entry the pooled
                spike times (`np.ndarray`) for one stimulus/repetition
                pair, ordered stimulus-major (s=0,r=0), (s=0,r=1), ...
              - labels: `np.ndarray` of shape (T,) with the stimulus
                index (0-indexed) of each trial.
    """
    _, S, R = spikes.shape
    T = S * R

    trials = [None] * T
    labels = np.zeros(T, dtype=int)

    idx = 0
    for s in range(S):
        for r in range(R):
            trials[idx] = pool_neurons(spikes, neurons, s, r)
            labels[idx] = s
            idx += 1

    return trials, labels


############################################################
# compute_population_distance_matrix
############################################################
def compute_population_distance_matrix(trials, Tmax, **kwargs):
    """ Computes the time-averaged pairwise SPIKE-distance matrix for a
    list of pooled trials, using PySpike's own
    :func:`pyspike.spike_distance_matrix`.

    :param trials: list of T pooled spike-time arrays (as returned by
                   :func:`.build_trials`), each an `np.ndarray`.
    :param Tmax: end of the recording interval; each trial is wrapped
                in a :class:`.SpikeTrain` spanning [0, Tmax].
    :param kwargs: forwarded as-is to
                   :func:`pyspike.spike_distance_matrix` (e.g. MRTS,
                   RI), so the SPIKE-distance variant (Classic / RI /
                   Adaptive / RIA) is chosen exactly as it would be for
                   any other PySpike call — no reinterpretation here.
    :returns: symmetric (T, T) `np.ndarray` of pairwise SPIKE-distances.
    """
    spike_trains = [SpikeTrain(t, [0.0, Tmax]) for t in trials]

    return spike_distance_matrix(spike_trains, **kwargs)


############################################################
# compute_discrimination_performance
############################################################
def compute_discrimination_performance(D, labels):
    """ Computes the discrimination performance :math:`P` (Eq. 1 in
    Satuvuori et al., 2018) from a pairwise distance matrix: the mean
    inter-stimulus distance minus the mean intra-stimulus distance.

    :param D: symmetric (T, T) distance matrix.
    :param labels: `np.ndarray` of shape (T,) with the stimulus index
                   of each trial (as returned by :func:`.build_trials`).
    :returns: discrimination performance :math:`P` (float).
    """
    labels = np.asarray(labels)
    T = len(labels)

    iu = np.triu_indices(T, k=1)
    same_stimulus = labels[iu[0]] == labels[iu[1]]

    intra = D[iu][same_stimulus]
    inter = D[iu][~same_stimulus]

    return np.mean(inter) - np.mean(intra)


############################################################
# evaluate_population
############################################################
def evaluate_population(spikes, neurons, Tmax, **kwargs):
    """ Computes the discrimination performance of a given subpopulation
    of neurons: pools their spike trains for every stimulus/repetition
    pair, computes the pairwise SPIKE-distance matrix, and derives the
    discrimination performance P.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param neurons: iterable of neuron indices (0-indexed) defining the
                    subpopulation to evaluate.
    :param Tmax: end of the recording interval.
    :param kwargs: forwarded to
                   :func:`.compute_population_distance_matrix` (e.g.
                   MRTS, RI).
    :returns: (P, D, labels) — discrimination performance, pairwise
              distance matrix, and stimulus label of each trial.
    """
    trials, labels = build_trials(spikes, neurons)
    D = compute_population_distance_matrix(trials, Tmax, **kwargs)
    P = compute_discrimination_performance(D, labels)
    return P, D, labels