# Module implementing the Labeled Line (LL) population coding analysis
# (Sec. 4.2 in Satuvuori et al., 2018), reimplemented following Delis
# et al. (2015) as noted in the original MATLAB code.
# Based on original MATLAB code (Maxime Beltoise / Eero Satuvuori),
# translated for PySpike.
# Distributed under the BSD License

import numpy as np
from scipy.stats import mannwhitneyu

from pyspike.algo.discrimination import build_trials, \
    compute_population_distance_matrix


############################################################
# performance_value_eero
############################################################
def performance_value_eero(D, S, R, alpha=1e-4):
    """ Computes, from a single neuron's pairwise SPIKE-distance matrix
    D, the discrimination matrix (Eq. 10), the raw mean-distance
    matrix, and the pairwise statistics needed for Eq. 11.

    Original name: PerformanceValue_Eero.m ('Eero' refers to Eero
    Satuvuori, the paper's original algorithm author) — kept as
    `performance_value_eero` (snake_case) for consistency with the
    rest of this module.

    :param D: (T, T) pairwise SPIKE-distance matrix for one neuron,
             trials ordered stimulus-major (as returned by
             :func:`.build_trials`), T = S*R.
    :param S: number of stimuli.
    :param R: number of repetitions per stimulus.
    :param alpha: significance level for the three Wilcoxon rank-sum
                 tests (default 1e-4, matching the original).
    :returns: (performance, S_matrix, R_matrix, distances, statistics)

        - performance: scalar, mean inter-stimulus distance minus mean
          intra-stimulus distance (same quantity as
          :func:`.compute_discrimination_performance`, computed here
          via the paper's explicit per-stimulus-pair bookkeeping).
        - S_matrix: (S, S) discrimination matrix (Eq. 10); 1 where the
          neuron discriminates the stimulus pair (at least one of the
          three tests is significant), 0 on the diagonal and where no
          test is significant.
        - R_matrix: (S, S) matrix of raw mean pairwise distances per
          stimulus pair (not used further downstream, kept for parity
          with the MATLAB output).
        - distances: (S, S) matrix of the maximum pairwise distance
          observed per stimulus pair (kept for parity, unused
          downstream in the original code too).
        - statistics: nested list of lists; statistics[sy][sx] (for
          sy <= sx) holds the raw list of pairwise distances
          contributing to that stimulus pair.
    """
    statistics = [[[] for _ in range(S)] for _ in range(S)]
    distances = np.zeros((S, S))

    intra_sum, intra_count = 0.0, 0
    inter_sum, inter_count = 0.0, 0

    for sy in range(S):
        for ry in range(R):
            y_index = sy * R + ry
            for sx in range(S):
                for rx in range(R):
                    x_index = sx * R + rx

                    if x_index > y_index:

                        d = D[x_index, y_index]

                        if sy == sx:
                            intra_sum += d
                            intra_count += 1
                        else:
                            inter_sum += d
                            inter_count += 1

                        if distances[sy, sx] < d:
                            distances[sy, sx] = d

                        statistics[sy][sx].append(d)

    R_matrix = np.zeros((S, S))
    for sy in range(S):
        for sx in range(sy, S):
            R_matrix[sy, sx] = np.mean(statistics[sy][sx])
            R_matrix[sx, sy] = R_matrix[sy, sx]

    # ------------------------------------------------------------
    # Identify groups (Eq. 10): three Wilcoxon rank-sum tests
    # ------------------------------------------------------------
    S_matrix = np.ones((S, S))

    for sy in range(S):
        for sx in range(sy, S):

            w1 = mannwhitneyu(statistics[sy][sx], statistics[sy][sy],
                              alternative='two-sided',
                              method='auto').pvalue < alpha
            w2 = mannwhitneyu(statistics[sy][sx], statistics[sx][sx],
                              alternative='two-sided',
                              method='auto').pvalue < alpha
            w3 = mannwhitneyu(statistics[sx][sx], statistics[sy][sy],
                              alternative='two-sided',
                              method='auto').pvalue < alpha

            if not (w1 or w2 or w3):
                S_matrix[sy, sx] = 0
                S_matrix[sx, sy] = 0

    if np.all(S_matrix == 1):
        S_matrix = np.zeros((S, S))

    performance = inter_sum / inter_count - intra_sum / intra_count

    return performance, S_matrix, R_matrix, distances, statistics


############################################################
# evaluate_LL_population
############################################################
def evaluate_LL_population(spikes, Tmax, alpha=1e-4, **kwargs):
    """ Evaluates the Labeled Line (LL) population coding hypothesis
    (Sec. 4.2 in Satuvuori et al., 2018): computes, for every neuron
    independently, its discrimination performance per stimulus pair
    (Eq. 10-11), then combines them into the population performance
    matrix (Eq. 12) and the optimal LL subpopulation (Eq. 14-15).

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param alpha: forwarded to :func:`.performance_value_eero`.
    :param kwargs: forwarded to
                   :func:`.compute_population_distance_matrix` (e.g.
                   MRTS, RI).
    :returns: dict with the keys:

        - DistanceMatrix / Discrimination / Performance / Mn: lists of
          length N (per-neuron matrices, see
          :func:`.performance_value_eero` and Eq. 11).
        - populationPerformance: (S, S) matrix P (Eq. 12).
        - bestNeuronMatrix: (S, S) int array; for each stimulus pair,
          the 0-indexed id of the best-discriminating neuron, or -1 if
          no neuron discriminates that pair (NB: unlike the MATLAB
          version, 0 is NOT the 'no neuron' sentinel here, since 0 is
          a valid 0-indexed neuron).
        - bestPopulation: sorted list of neuron indices (0-indexed)
          forming the optimal LL subpopulation (Eq. 14).
        - bestP: overall LL discrimination performance (Eq. 15).
    """
    num_neurons, num_stimuli, num_repetitions = spikes.shape

    distance_matrices = [None] * num_neurons
    discrimination_matrices = [None] * num_neurons
    performance_matrices = [None] * num_neurons
    mn_matrices = [None] * num_neurons

    for n in range(num_neurons):

        trials, _ = build_trials(spikes, [n])
        D = compute_population_distance_matrix(trials, Tmax, **kwargs)
        distance_matrices[n] = D

        _, S_matrix, R_matrix, _, statistics = performance_value_eero(
            D, num_stimuli, num_repetitions, alpha=alpha)

        discrimination_matrices[n] = S_matrix
        performance_matrices[n] = R_matrix

        Mn = np.zeros((num_stimuli, num_stimuli))
        for s1 in range(num_stimuli):
            for s2 in range(s1, num_stimuli):
                inter = np.mean(statistics[s1][s2])
                intra = np.mean(statistics[s1][s1] + statistics[s2][s2])
                value = S_matrix[s1, s2] * (inter - intra)
                Mn[s1, s2] = value
                Mn[s2, s1] = value

        mn_matrices[n] = Mn

    # ==================================================================
    # Population performance matrix (Eq. 12) and best-neuron matrix (Eq. 13)
    # ==================================================================
    p_population = np.zeros((num_stimuli, num_stimuli))
    best_neuron = np.full((num_stimuli, num_stimuli), -1, dtype=int)

    for s1 in range(num_stimuli):
        for s2 in range(s1, num_stimuli):

            best_value = 0.0
            best_index = -1

            for n in range(num_neurons):
                value = mn_matrices[n][s1, s2]
                if value > best_value:
                    best_value = value
                    best_index = n

            p_population[s1, s2] = best_value
            p_population[s2, s1] = best_value
            best_neuron[s1, s2] = best_index
            best_neuron[s2, s1] = best_index

    # ==================================================================
    # Optimal LL population (Eq. 14) / Global LL performance (Eq. 15)
    # ==================================================================
    best_population = sorted({n for n in best_neuron.ravel() if n >= 0})
    p_ll = np.mean(p_population[p_population > 0])

    return {
        'DistanceMatrix': distance_matrices,
        'Discrimination': discrimination_matrices,
        'Performance': performance_matrices,
        'Mn': mn_matrices,
        'populationPerformance': p_population,
        'bestNeuronMatrix': best_neuron,
        'bestPopulation': best_population,
        'bestP': p_ll,
    }