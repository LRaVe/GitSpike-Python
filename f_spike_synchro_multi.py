import numpy as np
from f_spike_synchro import f_spike_synchro


def f_spike_synchro_multi(st, t_min, t_max):
    """
    Multivariate SPIKE-Synchronization following equations (17-19) from the paper.
    
    Args:
        st: List of spike train arrays. Each element is a spike train for one neuron.
        t_min: Minimum time for analysis window
        t_max: Maximum time for analysis window
    
    Returns:
        C_matrix: n_trains x n_trains matrix of pairwise coincidence values
        C_global: Global SPIKE-Synchronization index (mean of upper triangle)
    """
    n_trains = len(st)
    C_matrix = np.zeros((n_trains, n_trains))
    
    # Compute pairwise coincidence matrices for all pairs of spike trains
    for i in range(n_trains):
        for j in range(n_trains):
            if i == j:
                C_matrix[i, j] = 1  # Perfect synchronization with itself
                continue  # Skip self-comparison
            
            # Compute coincidence between train i and train j
            C_ij, times_ij = f_spike_synchro(st[i], st[j], t_min, t_max)
            
            # Update C_matrix
            if len(times_ij) > 0:
                C_matrix[i, j] = np.sum(C_ij) / len(times_ij)  # Average over spikes
            else:
                C_matrix[i, j] = 0
    
    # Calculate global SPIKE-Synchronization index C_global as the mean of 
    # the upper triangle of C_matrix (excluding diagonal)
    upper_triangle_indices = np.triu_indices(n_trains, k=1)
    upper_triangle_values = C_matrix[upper_triangle_indices]
    
    if len(upper_triangle_values) > 0:
        C_global = np.mean(upper_triangle_values)
    else:
        C_global = 0
    
    return C_matrix, C_global
