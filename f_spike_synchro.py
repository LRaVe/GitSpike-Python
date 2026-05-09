import numpy as np


def f_spike_synchro(spike_train1, spike_train2, t_min, t_max):
    """
    Given two spike trains, this function calculates the coincidence of spikes between them.
    Uses a greedy matching algorithm to pair spikes based on minimum distance.
    
    Args:
        spike_train1: Array of spike times for first neuron
        spike_train2: Array of spike times for second neuron
        t_min: Minimum time for analysis window
        t_max: Maximum time for analysis window
    
    Returns:
        C: Coincidence array (1 if matched and within tau, 0 otherwise)
        spike_times: Corresponding spike times for each coincidence value
    """
    # Ensure inputs are numpy arrays and column vectors
    spike_train1 = np.atleast_1d(np.asarray(spike_train1)).flatten()
    spike_train2 = np.atleast_1d(np.asarray(spike_train2)).flatten()
    
    # Slice spikes within time window
    spike_train1_sliced = spike_train1[(spike_train1 >= t_min) & (spike_train1 <= t_max)]
    spike_train2_sliced = spike_train2[(spike_train2 >= t_min) & (spike_train2 <= t_max)]
    
    # Initialize tracking matrices
    n1 = len(spike_train1_sliced)
    n2 = len(spike_train2_sliced)
    
    if n1 == 0 and n2 == 0:
        return np.array([]), np.array([])
    
    # Go through the spikes in train1 and find the closest spike in train2
    # using the adaptive tau for coincidence detection
    C = np.zeros(n1)  # Initialize coincidence vector
    spike_times = spike_train1_sliced.copy()  # Corresponding spike times for C
    
    for i in range(n1):
        spike1 = spike_train1_sliced[i]
        tau1 = f_interval(spike_train1_sliced, spike1, t_min, t_max)
        
        for j in range(n2):
            spike2 = spike_train2_sliced[j]
            tau2 = f_interval(spike_train2_sliced, spike2, t_min, t_max)
            
            if f_in_interval(spike1, spike2, tau1, tau2):
                C[i] = 1  # Mark as coincident
                break  # Move to next spike in train1 after finding a match
            else:
                C[i] = 0  # Not coincident
    
    return C, spike_times


def f_interval(spike_train, spike, t_min, t_max):
    """
    Calculate tau for adaptive coincidence detection.
    tau = min(forward_ISI, backward_ISI) / 2
    Using eq. (15) and (16) for SPIKE-Synchronization
    
    Args:
        spike_train: Array of spike times
        spike: The spike time to analyze
        t_min: Minimum time for analysis window
        t_max: Maximum time for analysis window
    
    Returns:
        min_interval: The calculated tau value
    """
    # Find the index of the spike in the spike train
    spike_index = np.where(np.abs(spike_train - spike) < 1e-10)[0]
    
    if len(spike_index) == 0:
        return 0
    
    spike_index = spike_index[0]
    
    # Distance to previous spike
    if spike_index > 0:
        prev_dist = spike_train[spike_index] - spike_train[spike_index - 1]
    else:
        # For first spike: use ISI to next spike (or time window edge)
        if spike_index < len(spike_train) - 1:
            prev_dist = spike_train[spike_index + 1] - spike_train[spike_index]
        else:
            prev_dist = t_max - t_min  # Single spike fallback
    
    # Distance to next spike
    if spike_index < len(spike_train) - 1:
        next_dist = spike_train[spike_index + 1] - spike_train[spike_index]
    else:
        # For last spike: use ISI to previous spike (or time window edge)
        if spike_index > 0:
            next_dist = spike_train[spike_index] - spike_train[spike_index - 1]
        else:
            next_dist = t_max - t_min  # Single spike fallback
    
    min_interval = min(prev_dist, next_dist) / 2
    return min_interval


def f_in_interval(spike1, spike2, tau1, tau2):
    """
    Adaptive coincidence detection: eq. (15) and (16)
    tau_ij = min(tau_i, tau_j)
    Coincident if |spike1 - spike2| < tau_ij
    
    Args:
        spike1: First spike time
        spike2: Second spike time
        tau1: Tau value for first spike
        tau2: Tau value for second spike
    
    Returns:
        check: 1 if coincident, 0 otherwise
    """
    tau_ij = min(tau1, tau2)
    distance = abs(spike1 - spike2)
    
    if distance < tau_ij:
        return 1
    else:
        return 0
