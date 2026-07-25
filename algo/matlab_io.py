# Module for loading spike train datasets exported from MATLAB (.mat
# files), for cross-validation against the original MATLAB pipeline.
# Distributed under the BSD License

import numpy as np
import scipy.io as sio

from pyspike import SpikeTrain


############################################################
# load_spikes_from_mat
############################################################
def load_spikes_from_mat(path, Tmax=None, spikes_var='spikes'):
    """ Loads a (N, S, R) spike dataset saved from MATLAB (e.g. a
    workspace dump containing a `spikes` cell array, as produced by
    generate_SP_dataset.m / generate_LL_dataset.m) and converts it
    into the same format as :func:`.generate_SP_dataset` /
    :func:`.generate_LL_dataset`: a numpy object array of shape
    (N, S, R) of :class:`.SpikeTrain`.

    :param path: path to the .mat file.
    :param Tmax: end of the recording interval, used as the edge of
                every :class:`.SpikeTrain`. If None, this function
                tries to find it automatically in a `paramsLL` or
                `params` struct saved in the same file (as MATLAB's
                Main.m workspace does); raises if neither is found.
    :param spikes_var: name of the MATLAB variable holding the
                       (N, S, R) spikes cell array (default 'spikes').
    :returns: numpy object array of shape (N, S, R) of
              :class:`.SpikeTrain`.

    .. note:: Only classic (v5/v6/v7) .mat files are supported (via
       `scipy.io.loadmat`). Files saved with MATLAB's `-v7.3` flag
       (HDF5-based) will raise a `NotImplementedError` from scipy and
       need `h5py` / `mat73` instead.
    """
    data = sio.loadmat(path, simplify_cells=True)

    if spikes_var not in data:
        available = [k for k in data if not k.startswith('__')]
        raise KeyError(
            f"Variable '{spikes_var}' not found in {path}. "
            f"Available variables: {available}")

    raw_spikes = data[spikes_var]

    if Tmax is None:
        for params_var in ('paramsLL', 'params'):
            if params_var in data and 'Tmax' in data[params_var]:
                Tmax = float(data[params_var]['Tmax'])
                break
        if Tmax is None:
            raise ValueError(
                "Tmax not given and could not be found automatically "
                "in 'paramsLL'/'params' in the .mat file; pass it "
                "explicitly.")

    N, S, R = raw_spikes.shape
    spikes = np.empty((N, S, R), dtype=object)

    for n in range(N):
        for s in range(S):
            for r in range(R):
                t = np.atleast_1d(
                    np.asarray(raw_spikes[n, s, r], dtype=float)).ravel()
                spikes[n, s, r] = SpikeTrain(np.sort(t), [0.0, Tmax])

    return spikes