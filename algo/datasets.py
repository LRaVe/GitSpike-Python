# Module for generating simulated datasets (Summed Population / Labeled Line)
# for the discriminative subpopulation analysis.
# Based on original MATLAB code, translated for PySpike.
# Distributed under the BSD License

import numpy as np
from pyspike import SpikeTrain


############################################################
# generate_SP_dataset
############################################################
def generate_SP_dataset(params):
    """ Generates a simulated Summed Population (SP) dataset, following
    the setup described in Satuvuori et al. (2018).

    The population consists of three groups of neurons:

      - 'Coll' neurons (the first `c` neurons): code the stimuli
        collectively. For each stimulus a single pooled spike train is
        generated, and at every repetition its spikes are randomly
        redistributed among the `c` coding neurons.
      - 'Indi' neurons (the next `nIndi` neurons): code each stimulus
        individually and reliably, with a small Gaussian jitter added
        at every repetition.
      - Non-coding neurons (the remaining neurons): fire as
        independent Poisson background noise, regardless of stimulus
        or repetition.

    :param params: dict with the keys:

        - N: total number of neurons.
        - c: number of collectively coding ('Coll') neurons.
        - nIndi: number of individually coding ('Indi') neurons.
        - S: number of stimuli.
        - R: number of repetitions per stimulus.
        - Tmax: duration of the recording interval [0, Tmax].
        - rate: firing rate (spikes/s) of a single neuron.
        - indiJitter: std (in s) of the Gaussian jitter applied to the
          'Indi' neurons at every repetition.

    :returns: numpy object array `spikes` of shape (N, S, R), each
              entry a :class:`.SpikeTrain` for neuron n, stimulus s,
              repetition r (0-indexed, unlike the original 1-indexed
              MATLAB code).
    """
    N = params['N']
    c = params['c']
    nIndi = params['nIndi']
    S = params['S']
    R = params['R']
    Tmax = params['Tmax']
    rate = params['rate']
    indi_jitter = params['indiJitter']

    spikes = np.empty((N, S, R), dtype=object)

    # =====================================================
    # COLL neurons: collectively coding subpopulation
    # =====================================================
    pooled_rate = c * rate

    for s in range(S):

        n_spikes = np.random.poisson(pooled_rate * Tmax)
        pooled_train = np.sort(np.random.rand(n_spikes) * Tmax)

        for r in range(R):

            # randomly assign each pooled spike to one of the c neurons
            assignment = np.random.randint(0, c, n_spikes)

            for n in range(c):
                spikes[n, s, r] = SpikeTrain(pooled_train[assignment == n],
                                             [0.0, Tmax])

    # =====================================================
    # INDI neurons: individually and reliably coding neurons
    # =====================================================
    for n in range(c, c + nIndi):

        # one template spike train per stimulus, shared across repetitions
        template = []
        for s in range(S):
            n_spikes = np.random.poisson(rate * Tmax)
            template.append(np.sort(np.random.rand(n_spikes) * Tmax))

        for s in range(S):
            for r in range(R):

                train = template[s] + indi_jitter * np.random.randn(
                    len(template[s]))
                train = np.clip(train, 0.0, Tmax)

                spikes[n, s, r] = SpikeTrain(np.sort(train), [0.0, Tmax])

    # =====================================================
    # NON CODING neurons: pure background noise
    # =====================================================
    for n in range(c + nIndi, N):
        for s in range(S):
            for r in range(R):

                n_noise = np.random.poisson(rate * Tmax)
                spikes[n, s, r] = SpikeTrain(
                    np.sort(np.random.rand(n_noise) * Tmax), [0.0, Tmax])

    return spikes


############################################################
# generate_LL_dataset
############################################################
def generate_LL_dataset(params):
    """ Generates a simulated Labeled Line (LL) dataset, following the
    setup described in Satuvuori et al. (2018).

    Each neuron is either sensitive or insensitive to each stimulus, as
    given by the response matrix. For a (neuron, stimulus) pair the
    neuron responds to, a spike train template is generated once
    (shared across all stimuli it responds to, or specific to each
    stimulus, depending on `sameResponse`) and copied at every
    repetition with jitter noise added. For a (neuron, stimulus) pair
    the neuron does not respond to, every repetition is independent
    Poisson background noise.

    :param params: dict with the keys:

        - N: number of neurons.
        - S: number of stimuli.
        - R: number of repetitions per stimulus.
        - mode: 'structured' to provide `responseMatrix` explicitly, or
          'random' to generate it using `connectionProbability`.
        - responseMatrix: (mode='structured') N x S boolean/0-1 array.
        - connectionProbability: (mode='random', default 0.4)
          probability that a neuron responds to a given stimulus.
        - meanRate: baseline firing rate (spikes/s).
        - Tmax: duration of the recording interval [0, Tmax].
        - jitter: base amplitude (in s) of the jitter noise.
        - jitterIntensity: array of length N, per-neuron multiplier of
          `jitter` (controls each neuron's reliability).
        - sameResponse: (optional, default all True) boolean array of
          length N. If True, a single template is shared across all
          stimuli the neuron responds to; if False, a new template is
          drawn independently for every stimulus.

    :returns: (spikes, response_matrix)

              - spikes: numpy object array of shape (N, S, R), each
                entry a :class:`.SpikeTrain` (0-indexed).
              - response_matrix: the (possibly randomly generated)
                N x S response matrix.
    """
    N = params['N']
    S = params['S']
    R = params['R']

    # ------------------------------------------------------------------
    # Response matrix
    # ------------------------------------------------------------------
    mode = params['mode'].lower()

    if mode == 'structured':

        response_matrix = np.asarray(params['responseMatrix'])

        if response_matrix.shape != (N, S):
            raise ValueError('responseMatrix must be of size N x S.')

    elif mode == 'random':

        p = params.get('connectionProbability', 0.4)

        response_matrix = np.random.rand(N, S) < p

        # ensure each neuron responds to at least one stimulus
        for n in range(N):
            if not np.any(response_matrix[n, :]):
                response_matrix[n, np.random.randint(S)] = True

    else:
        raise ValueError('Unknown mode.')

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    spikes = np.empty((N, S, R), dtype=object)

    mean_rate = params['meanRate']
    Tmax = params['Tmax']
    jitter = params['jitter']
    jitter_intensity = params['jitterIntensity']

    same_response = np.asarray(
        params.get('sameResponse', np.ones(N, dtype=bool)),
        dtype=bool).ravel()

    if len(same_response) != N:
        raise ValueError('sameResponse must have length N.')

    for n in range(N):

        # ================================================================
        # Case 1: a single template shared among all stimuli
        # ================================================================
        if same_response[n]:
            n_spikes_template = np.random.poisson(mean_rate * Tmax)
            shared_template = np.sort(np.random.rand(n_spikes_template) * Tmax)

        for s in range(S):

            if response_matrix[n, s]:

                # ========================================================
                # Case 2: a template specific to each stimulus
                # ========================================================
                if not same_response[n]:
                    n_spikes_template = np.random.poisson(mean_rate * Tmax)
                    shared_template = np.sort(
                        np.random.rand(n_spikes_template) * Tmax)

                for r in range(R):

                    stim_shift = s * 0.002 * Tmax

                    noise = ((2 * np.random.rand(len(shared_template)) - 1)
                            * jitter * jitter_intensity[n])

                    trial = shared_template + stim_shift + noise
                    trial = trial[(trial >= 0) & (trial <= Tmax)]

                    spikes[n, s, r] = SpikeTrain(np.sort(trial), [0.0, Tmax])

            else:

                # ========================================================
                # No response: background noise
                # ========================================================
                for r in range(R):

                    n_spikes = np.random.poisson(mean_rate * Tmax)
                    spikes[n, s, r] = SpikeTrain(
                        np.sort(np.random.rand(n_spikes) * Tmax), [0.0, Tmax])

    return spikes, response_matrix