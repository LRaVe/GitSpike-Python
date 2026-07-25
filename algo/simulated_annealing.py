# Module implementing the simulated annealing search for the most
# discriminative SP subpopulation, with caching of evaluated
# subpopulations. Based on original MATLAB code (Maxime Beltoise),
# translated for PySpike.
#
# NB: this is unrelated to PySpike's existing simulated annealing for
# the optimal spike train order (pyspike.spike_directionality).
# Distributed under the BSD License

import numpy as np

from pyspike.algo.discrimination import evaluate_population


############################################################
# EvaluationCache
############################################################
class EvaluationCache:
    """ Memoizes :func:`.evaluate_population` results per subpopulation,
    keyed by the sorted tuple of neuron indices. Used by
    :func:`.evaluate_population_cached`.

    Replaces the MATLAB struct of `cache.data` (a `containers.Map`)
    plus `cache.hits`/`cache.misses`. Since this is a regular mutable
    Python object (unlike a MATLAB struct), it does not need to be
    returned/reassigned after every call — mutations are visible
    everywhere the same instance is used.
    """

    def __init__(self):
        self.data = {}
        self.hits = 0
        self.misses = 0


############################################################
# evaluate_population_cached
############################################################
def evaluate_population_cached(spikes, neurons, Tmax, cache, **kwargs):
    """ Cached version of :func:`.evaluate_population`: returns the
    memoized (P, D, labels) for a given subpopulation if already
    computed, otherwise computes and stores it.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param neurons: iterable of neuron indices (0-indexed).
    :param Tmax: end of the recording interval.
    :param cache: :class:`.EvaluationCache` instance, mutated in place.
    :param kwargs: forwarded to :func:`.evaluate_population`.
    :returns: (P, D, labels), as :func:`.evaluate_population`.
    """
    key = tuple(sorted(neurons))

    if key in cache.data:
        cache.hits += 1
        return cache.data[key]

    cache.misses += 1
    result = evaluate_population(spikes, neurons, Tmax, **kwargs)
    cache.data[key] = result

    return result


############################################################
# random_neighbor
############################################################
def random_neighbor(population, N):
    """ Draws a random neighboring subpopulation of `population` by
    adding or removing a single, randomly chosen neuron (Sec. 4.1.2.3
    in Satuvuori et al., 2018, Eq. 8-9 context). Addition/removal is
    chosen with equal probability, except at the boundaries (a
    singleton population must grow, the full population must shrink).

    :param population: iterable of neuron indices (0-indexed).
    :param N: total number of neurons.
    :returns: list, the neighboring subpopulation.
    """
    candidate = list(population)
    n_pop = len(candidate)

    # ---------------------------------
    # full population
    # ---------------------------------
    if n_pop == N:
        idx = np.random.randint(n_pop)
        del candidate[idx]
        return candidate

    # ---------------------------------
    # population size 1
    # ---------------------------------
    if n_pop == 1:
        missing = [n for n in range(N) if n not in population]
        candidate.append(missing[np.random.randint(len(missing))])
        return sorted(candidate)

    # ---------------------------------
    # add/remove 50-50
    # ---------------------------------
    if np.random.rand() < 0.5:
        # remove
        idx = np.random.randint(n_pop)
        del candidate[idx]
    else:
        # add
        missing = [n for n in range(N) if n not in population]
        candidate.append(missing[np.random.randint(len(missing))])
        candidate = sorted(candidate)

    return candidate


############################################################
# metropolis_acceptance
############################################################
def metropolis_acceptance(P_candidate, P_current, T):
    """ Metropolis acceptance criterion (Eq. 8 in Satuvuori et al.,
    2018): always accept an improving move, accept a worsening move
    with probability :math:`\\exp(-\\Delta P / T)`.

    :param P_candidate: discrimination performance of the candidate.
    :param P_current: discrimination performance of the current
                      population.
    :param T: current (pseudo-)temperature.
    :returns: bool, whether to accept the candidate.
    """
    if P_candidate > P_current:
        return True

    delta_p = P_current - P_candidate
    q = np.exp(-delta_p / T)

    return np.random.rand() < q


############################################################
# initialize_temperature
############################################################
def initialize_temperature(spikes, Tmax, N0, **kwargs):
    """ Estimates the initial temperature T0 (Eq. 9 in Satuvuori et
    al., 2018) from a path of N0 random test steps, so that downhill
    moves are accepted with ~95% likelihood at the start of the search.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param N0: number of random test steps used to estimate T0.
    :param kwargs: forwarded to :func:`.evaluate_population`.
    :returns: float, the initial temperature T0.
    """
    N = spikes.shape[0]

    pop_size = np.random.randint(1, N + 1)
    pop = list(np.random.choice(N, size=pop_size, replace=False))

    P_prev, _, _ = evaluate_population(spikes, pop, Tmax, **kwargs)

    delta_p = np.zeros(N0)

    for k in range(N0):
        pop = random_neighbor(pop, N)
        P_new, _, _ = evaluate_population(spikes, pop, Tmax, **kwargs)
        delta_p[k] = abs(P_new - P_prev)
        P_prev = P_new

    mean_delta = np.mean(delta_p)

    return -mean_delta / np.log(0.95)


############################################################
# simulated_annealing
############################################################
def simulated_annealing(spikes, Tmax, params_sa, verbose=False, **kwargs):
    """ Simulated annealing search for the most discriminative SP
    subpopulation (Sec. 4.1.2.3 in Satuvuori et al., 2018): starting
    from a random subpopulation, repeatedly proposes a random
    neighboring subpopulation and accepts/rejects it via the Metropolis
    criterion, with a step-wise cooling schedule and reannealing when a
    temperature plateau converges to a value worse than the best found.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param Tmax: end of the recording interval.
    :param params_sa: dict with the keys:

        - N0: number of random steps used to estimate the initial
          temperature (see :func:`.initialize_temperature`).
        - steps: number of test steps per temperature plateau.
        - coolingFactor: multiplicative cooling factor (e.g. 0.95).
        - codingNeurons: iterable of ground-truth coding neuron
          indices (0-indexed), used only to track population
          composition in the search history (not used by the search
          itself).

    :param verbose: if True, prints progress and a final summary.
    :param kwargs: forwarded to :func:`.evaluate_population` (via the
                   cached wrapper).
    :returns: dict with the keys:

        - bestPopulation: sorted list of neuron indices of the best
          subpopulation found.
        - bestP: its discrimination performance.
        - history: dict of `np.ndarray` (one entry per search step):
          'P', 'bestP', 'size', 'temperature', 'nCoding', 'nNonCoding'.
        - iterations: total number of search steps.
        - cacheHits / cacheMisses: cache statistics.
        - acceptanceRate: fraction of proposed moves accepted.
        - hitRate: cache hit rate.
    """
    N = spikes.shape[0]

    cache = EvaluationCache()

    accepted_moves = 0
    total_moves = 0
    accepted_better = 0
    accepted_worse = 0

    # =====================================
    # initial temperature
    # =====================================
    T0 = initialize_temperature(spikes, Tmax, params_sa['N0'], **kwargs)

    if verbose:
        print(f'Initial temperature T0 = {T0:.4f}')

    T = T0

    # =====================================
    # initial population
    # =====================================
    pop_size = np.random.randint(1, N + 1)
    current_pop = sorted(
        np.random.choice(N, size=pop_size, replace=False).tolist())

    P_current, _, _ = evaluate_population_cached(
        spikes, current_pop, Tmax, cache, **kwargs)

    # =====================================
    # best population found
    # =====================================
    best_pop = list(current_pop)
    best_p = P_current

    delta_p_history = []

    history = {
        'P': [], 'bestP': [], 'size': [], 'temperature': [],
        'nCoding': [], 'nNonCoding': [],
    }

    steps_per_temp = params_sa['steps']
    coding_neurons = set(params_sa['codingNeurons'])

    # =====================================
    # main loop
    # =====================================
    while True:

        population_changed = False

        for k in range(steps_per_temp):

            candidate = random_neighbor(current_pop, N)

            P_candidate, _, _ = evaluate_population_cached(
                spikes, candidate, Tmax, cache, **kwargs)

            accept = metropolis_acceptance(P_candidate, P_current, T)

            total_moves += 1

            if accept:

                accepted_moves += 1

                if P_candidate > P_current:
                    accepted_better += 1
                else:
                    accepted_worse += 1

                if sorted(candidate) != sorted(current_pop):
                    population_changed = True

                current_pop = candidate
                P_current = P_candidate

            # best found
            if P_current > best_p:
                best_p = P_current
                best_pop = list(current_pop)

            # search history
            history['P'].append(P_current)
            history['bestP'].append(best_p)
            history['size'].append(len(current_pop))
            history['temperature'].append(T)
            delta_p_history.append(abs(P_candidate - P_current))

            n_coding = len(coding_neurons.intersection(current_pop))
            history['nCoding'].append(n_coding)
            history['nNonCoding'].append(len(current_pop) - n_coding)

        # =====================================
        # convergence
        # =====================================
        if not population_changed:

            # reannealing
            if best_p > P_current:

                if verbose:
                    print('Reannealing...')

                current_pop = list(best_pop)
                P_current = best_p
                T = T0

                continue

            else:
                break

        # cooling
        T = T * params_sa['coolingFactor']

    # =====================================
    # output
    # =====================================
    result = {
        'bestPopulation': sorted(best_pop),
        'bestP': best_p,
        'history': {k: np.array(v) for k, v in history.items()},
        'iterations': len(history['P']),
        'cacheHits': cache.hits,
        'cacheMisses': cache.misses,
        'acceptanceRate': accepted_moves / total_moves,
        'hitRate': cache.hits / (cache.hits + cache.misses),
    }

    if verbose:
        n_scored = accepted_better + accepted_worse
        worse_pct = 100 * accepted_worse / n_scored if n_scored > 0 else float('nan')

        print()
        print(f"Iterations           : {result['iterations']}")
        print(f"Acceptance rate      : {100 * result['acceptanceRate']:.1f} %")
        print(f"Accepted better      : {accepted_better}")
        print(f"Accepted worse       : {accepted_worse}")
        print(f"Worse acceptance     : {worse_pct:.1f} %")
        print(f"Cache hits           : {cache.hits}")
        print(f"Cache misses         : {cache.misses}")
        print(f"Unique populations   : {cache.misses}")
        print(f"Cache hit rate       : {100 * result['hitRate']:.1f} %")
        print(f"Mean |\u0394P| = {np.mean(delta_p_history):.6f}")
        print()

    return result