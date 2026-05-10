"""Shared helper functions used by the spike-order packages.

This module contains the functions for adding auxiliary spikes and
computing coincidence windows.
"""

import numpy as np
from typing import Tuple


# ======================================
# ====== Auxiliary spike handling ======
# ======================================

def process_single_train(train: np.ndarray, t_min: float, t_max: float) -> Tuple[np.ndarray,int,int]:
    aux_begin = 0
    aux_end = 0
    train = np.unique(np.asarray(train))
    if train.size == 0:
        return train, aux_begin, aux_end

    # Add an auxiliary spike at the beginning if needed
    if train[0] > t_min:
        if train.size >= 2:
            aux = train[0] - max(train[0]-t_min, train[1]-train[0])
        else:
            aux = t_min
        train = np.concatenate(([aux], train))
        aux_begin = 1

    # Add an auxiliary spike at the end if needed
    if train[-1] < t_max:
        if train.size >= 2:
            aux = train[-1] + max(t_max-train[-1], train[-1]-train[-2])
        else:
            aux = t_max
        train = np.concatenate((train, [aux]))
        aux_end = 1

    return train, aux_begin, aux_end


def add_auxiliary_spikes(spikes, t_min, t_max):
    if isinstance(spikes, (list, tuple)):
        aux_begin = [0]*len(spikes)
        aux_end = [0]*len(spikes)
        out = []
        for i, tr in enumerate(spikes):
            new_tr, b, e = process_single_train(np.asarray(tr), t_min, t_max)
            out.append(new_tr)
            aux_begin[i] = b
            aux_end[i] = e
        return out, aux_begin, aux_end

    # Single train input
    train, aux_begin, aux_end = process_single_train(np.asarray(spikes), t_min, t_max)
    return train, aux_begin, aux_end


# ============================================
# ====== Coincidence window computation ======
# ============================================

def coincidence_window(tmin, tmax, spikes, spike_ind1, spike_ind2, ind1, ind2):
    n = len(spikes)
    if spike_ind1>n or spike_ind2>n or spike_ind1<1 or spike_ind2<1:
        raise IndexError('Index out of bounds')

    s1 = np.asarray(spikes[spike_ind1-1])
    s2 = np.asarray(spikes[spike_ind2-1])

    if ind1<1 or ind1>len(s1) or ind2<1 or ind2>len(s2):
        raise IndexError('Index out of bounds')

    # Convert MATLAB-style one-based indices to Python zero-based indices
    i1 = ind1-1
    i2 = ind2-1

    prev1 = s1[i1]-tmin if i1==0 else s1[i1]-s1[i1-1]
    next1 = tmax-s1[i1] if i1==len(s1)-1 else s1[i1+1]-s1[i1]
    prev2 = s2[i2]-tmin if i2==0 else s2[i2]-s2[i2-1]
    next2 = tmax-s2[i2] if i2==len(s2)-1 else s2[i2+1]-s2[i2]

    window = min(prev1, next1, prev2, next2)/2.0
    return window
