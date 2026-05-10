import numpy as np
from spike_common import coincidence_window


def pairwise_train_order(tmin, tmax, spikes, spike_ind1, spike_ind2):
    n = len(spikes)
    if spike_ind1 > n or spike_ind2 > n or spike_ind1 < 1 or spike_ind2 < 1:
        raise IndexError('Index out of bounds')

    s1 = np.asarray(spikes[spike_ind1 - 1])
    s2 = np.asarray(spikes[spike_ind2 - 1])
    res1 = np.zeros(len(s1), dtype=float)
    res2 = np.zeros(len(s2), dtype=float)

    for i in range(len(s1)):
        for j in range(len(s2)):
            if abs(s1[i] - s2[j]) < coincidence_window(tmin, tmax, spikes, spike_ind1, spike_ind2, i + 1, j + 1):
                if s1[i] < s2[j]:
                    signValue = 1.0
                elif s1[i] > s2[j]:
                    signValue = -1.0
                else:
                    signValue = 0.0
                res1[i] = signValue
                res2[j] = signValue
                break
    return res1, res2
