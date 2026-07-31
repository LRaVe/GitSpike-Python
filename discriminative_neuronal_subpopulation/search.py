# Module implementing the search algorithms (brute force, gradient,
# and later simulated annealing) for the discriminative subpopulation
# analysis (Summed Population case).
# Based on original MATLAB code (Maxime Beltoise / Laure Wolff),
# translated for PySpike.
# Distributed under the BSD License

import numpy as np

from discriminative_neuronal_subpopulation.discrimination import evaluate_population
from discriminative_neuronal_subpopulation.datasets import generate_SP_dataset
from discriminative_neuronal_subpopulation.simulated_annealing import simulated_annealing


############################################################
# brute_force_search
############################################################
def brute_force_search(spikes, Tmax, verbose=False, **kwargs):
    """ Brute force search for the most discriminative SP subpopulation
    (Sec. 4.1.2.1 in Satuvuori et al., 2018): evaluates every possible
    non-empty subpopulation and returns the one with the highest
    discrimination performance.

    Guaranteed to find the global optimum, but the number of evaluated
    subpopulations grows as :math:`2^N - 1`, so this is only feasible
    for small N (aborts above N=20).

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param verbose: if True, prints progress and the final result.
    :param kwargs: forwarded to :func:`.evaluate_population`.
    :returns: dict with the keys:

        - bestSubpopulation: list of neuron indices (0-indexed) of the
          best subpopulation found.
        - bestPerformance: its discrimination performance.
        - historyPerf: `np.ndarray` of length 2**N-1, the performance
          of every evaluated subpopulation in enumeration order
          (binary mask 1 to 2**N-1).
    """
    N = spikes.shape[0]
    total_combinations = (2 ** N) - 1

    if N > 20:
        raise ValueError(
            f'Brute force aborted: N is too large ({N}). Reduce N '
            'between 10 and 20.')

    if verbose:
        print(f'-> Launching brute force by binary incrementation '
              f'({total_combinations} masks to evaluate...)')

    best_perf_overall = -np.inf
    best_subpop = []
    history_perf_brute = np.zeros(total_combinations)

    for i in range(1, total_combinations + 1):

        population = [n for n in range(N) if (i >> n) & 1]

        perf, _, _ = evaluate_population(spikes, population, Tmax, **kwargs)
        history_perf_brute[i - 1] = perf

        if perf > best_perf_overall:
            best_perf_overall = perf
            best_subpop = population

    if verbose:
        best_subpop_print = [n + 1 for n in best_subpop]
        print('\n================ BRUTE FORCE CONVERGED ================')
        print(f'Best combination found: {best_subpop_print}')
        print(f'Absolute maximum performance P = {best_perf_overall:.4f}')
        print('=======================================================')

    return {
        'bestSubpopulation': best_subpop,
        'bestPerformance': best_perf_overall,
        'historyPerf': history_perf_brute,
    }


############################################################
# bottom_up_search
############################################################
def bottom_up_search(spikes, Tmax, verbose=False, **kwargs):
    """ Bottom-up gradient (steepest-ascent) search for the most
    discriminative SP subpopulation (Sec. 4.1.2.2 in Satuvuori et al.,
    2018): starts empty and iteratively adds the neuron that improves
    the discrimination performance the most, until every neuron has
    been added.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param verbose: if True, prints progress at each step.
    :param kwargs: forwarded to :func:`.evaluate_population`.
    :returns: dict with the keys:

        - bestOrder: list of length N, neuron indices in the order
          they were added.
        - historyPerf: `np.ndarray` of length N, best performance
          reached at each step k (population of size k+1).
        - matrixGrid: (N, N) `np.ndarray`; matrixGrid[k, n] is the
          performance of adding neuron n at step k (NaN if n was
          already in the population before step k).
        - bestSubpopulation: prefix of `bestOrder` achieving the
          highest value in `historyPerf`.
        - bestPerformance: that highest value.
    """
    N = spikes.shape[0]

    best_order = []
    neurons_dispo = list(range(N))
    history_perf = np.zeros(N)
    matrix_grid = np.full((N, N), np.nan)
    current_pop = []

    for k in range(N):

        num_dispo = len(neurons_dispo)
        current_step_perf = np.full(num_dispo, -np.inf)

        for i in range(num_dispo):

            candidate = current_pop + [neurons_dispo[i]]
            P_candidate, _, _ = evaluate_population(spikes, candidate, Tmax,
                                                    **kwargs)

            current_step_perf[i] = P_candidate
            matrix_grid[k, neurons_dispo[i]] = P_candidate

        best_idx = int(np.argmax(current_step_perf))
        best_perf_step = current_step_perf[best_idx]
        best_neurone_step = neurons_dispo[best_idx]

        current_pop = current_pop + [best_neurone_step]
        best_order = list(current_pop)
        del neurons_dispo[best_idx]
        history_perf[k] = best_perf_step

        if verbose:
            print(f'Step k = {k + 1} | Adding neuron : {best_neurone_step + 1} '
                  f'| Performance P = {best_perf_step:.4f}')

    idx_max_absolu = int(np.argmax(history_perf))
    best_subpop = best_order[:idx_max_absolu + 1]

    return {
        'bestOrder': best_order,
        'historyPerf': history_perf,
        'matrixGrid': matrix_grid,
        'bestSubpopulation': best_subpop,
        'bestPerformance': history_perf[idx_max_absolu],
    }


############################################################
# top_down_search
############################################################
def top_down_search(spikes, Tmax, **kwargs):
    """ Top-down gradient (steepest-ascent) search for the most
    discriminative SP subpopulation (Sec. 4.1.2.2 in Satuvuori et al.,
    2018): starts from the full population and iteratively removes the
    neuron whose removal contributes least (i.e. yields the best
    remaining performance), until a single neuron remains.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param kwargs: forwarded to :func:`.evaluate_population`.
    :returns: dict with the keys:

        - populations: list of populations visited, from the full
          population (index 0) down to the singleton (last index).
        - P: `np.ndarray`, performance of each visited population.
        - removedNeuron: list, neuron removed at each step (length
          N-1).
        - candidateP: list (per step) of length-N arrays giving the
          performance of removing each currently-present neuron (NaN
          for neurons not in the current population).
        - candidatePop: list (per step) of length-N lists, the
          resulting candidate population per neuron (None if not
          applicable).
        - bestP / bestPopulation / bestStep: best performance found,
          the population achieving it, and its (0-indexed) step.
    """
    N = spikes.shape[0]
    current_pop = list(range(N))

    result = {
        'populations': [],
        'P': [],
        'removedNeuron': [],
        'candidateP': [],
        'candidatePop': [],
    }

    while len(current_pop) >= 1:

        P_current, _, _ = evaluate_population(spikes, current_pop, Tmax,
                                              **kwargs)
        result['populations'].append(list(current_pop))
        result['P'].append(P_current)

        if len(current_pop) == 1:
            break

        best_p = -np.inf
        best_neuron = None
        best_pop = None

        candidate_perf = np.full(N, np.nan)
        candidate_pops = [None] * N

        for k in range(len(current_pop)):

            candidate = current_pop[:k] + current_pop[k + 1:]
            P_candidate, _, _ = evaluate_population(spikes, candidate, Tmax,
                                                    **kwargs)

            candidate_perf[current_pop[k]] = P_candidate
            candidate_pops[current_pop[k]] = candidate

            if P_candidate > best_p:
                best_p = P_candidate
                best_neuron = current_pop[k]
                best_pop = candidate

        result['removedNeuron'].append(best_neuron)
        result['candidateP'].append(candidate_perf)
        result['candidatePop'].append(candidate_pops)

        current_pop = best_pop

    result['P'] = np.array(result['P'])
    idx = int(np.argmax(result['P']))

    result['bestP'] = result['P'][idx]
    result['bestPopulation'] = result['populations'][idx]
    result['bestStep'] = idx

    return result


############################################################
# run_complexity_benchmark
############################################################
def run_complexity_benchmark(neuron_counts, sp_params, sa_params, loop=3,
                             steps_per_neuron=5, verbose=True, **kwargs):
    """ Reproduces Fig. 8 of Satuvuori et al. (2018): compares the
    number of subpopulations evaluated by the different SP search
    algorithms as a function of the number of neurons N.

    The Bottom-Up/Top-Down and Brute Force counts are the theoretical,
    closed-form values (Eq. 6-7) — these algorithms are NOT actually
    run here, matching the original MATLAB benchmark (which only
    computes N*(N+1)/2 and 2**N-1 directly). Only Simulated Annealing
    is actually executed (`loop` times per neuron count, averaged),
    since its cost cannot be predicted in closed form.

    :param neuron_counts: iterable of population sizes N to test.
    :param sp_params: dict of SP dataset parameters (must provide 'S',
                      'R', 'Tmax', 'rate'; must NOT include 'N' or
                      'c', which are set per neuron count — 'c' is
                      fixed to N//2, and 'nIndi'/'indiJitter' default
                      to 0 unless already present).
    :param sa_params: dict of simulated annealing parameters, must
                      provide 'N0' and 'coolingFactor' ('steps' and
                      'codingNeurons' are set automatically per N).
    :param loop: number of independent SA runs averaged per neuron
                count (default 3).
    :param steps_per_neuron: SA steps per temperature plateau, scaled
                             as `steps_per_neuron * N` (default 5,
                             matching the original `5*params.N`).
    :param verbose: if True, prints progress.
    :param kwargs: forwarded to :func:`.simulated_annealing` (e.g.
                   MRTS, RI). NB: the original script used the
                   Adaptive SPIKE-distance (Distances=[0 0 1 0]); pass
                   the equivalent PySpike keyword here if you want to
                   match that, otherwise PySpike's defaults apply.
    :returns: dict with the keys 'neuronCounts', 'greedy',
              'saIterations', 'saUnique', 'bruteForce' — each an
              `np.ndarray` aligned with `neuron_counts`.
    """
    neuron_counts = np.asarray(neuron_counts)

    eval_greedy = np.zeros(len(neuron_counts))
    eval_sa = np.zeros(len(neuron_counts))
    eval_sa_unique = np.zeros(len(neuron_counts))
    eval_brute = np.zeros(len(neuron_counts))

    if verbose:
        print('Starting Evaluation Count Benchmark...')

    for idx, N in enumerate(neuron_counts):
        N = int(N)

        params = dict(sp_params)
        params['N'] = N
        params['c'] = N // 2
        params.setdefault('nIndi', 0)
        params.setdefault('indiJitter', 0)

        if verbose:
            print('\n---------------------------------')
            print(f'Testing N = {N}')

        spikes = generate_SP_dataset(params)

        # Bottom-Up / Top-Down: theoretical count only (Eq. 7)
        if verbose:
            print('Evaluating Bottom-up/ Top-down...')
        eval_greedy[idx] = N * (N + 1) / 2

        # Simulated Annealing: actually run `loop` times
        if verbose:
            print('Evaluating Simulated Annealing...')
            print('Running Simulated Annealing...')

        params_sa = dict(sa_params)
        params_sa['steps'] = steps_per_neuron * N
        params_sa['codingNeurons'] = list(range(params['c']))

        iterations = np.zeros(loop)
        unique_pop = np.zeros(loop)

        for k in range(loop):
            SA = simulated_annealing(spikes, params['Tmax'], params_sa,
                                    **kwargs)
            iterations[k] = SA['iterations']
            unique_pop[k] = SA['cacheMisses']

        eval_sa[idx] = iterations.mean()
        eval_sa_unique[idx] = unique_pop.mean()

        # Brute Force: theoretical count only (Eq. 6)
        if verbose:
            print('Evaluating Brute Force...')
        eval_brute[idx] = 2 ** N - 1

    if verbose:
        print('\nBenchmark completed successfully!')

    return {
        'neuronCounts': neuron_counts,
        'greedy': eval_greedy,
        'saIterations': eval_sa,
        'saUnique': eval_sa_unique,
        'bruteForce': eval_brute,
    }