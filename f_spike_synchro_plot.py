import matplotlib.pyplot as plt
import numpy as np


def f_spike_synchro_plot(C_matrix, C_global):
    """
    Plotting function for SPIKE-Synchronization results.
    This function visualizes the results of the SPIKE-Synchronization analysis.
    
    Args:
        C_matrix: Pairwise coincidence matrix (n_trains x n_trains)
        C_global: Global SPIKE-Synchronization index (scalar)
    
    Returns:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot pairwise coincidence matrix as heatmap
    im = ax.imshow(C_matrix, cmap='jet', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coincidence Value', rotation=270, labelpad=15)
    
    # Labels and title
    ax.set_xlabel('Spike Train Index')
    ax.set_ylabel('Spike Train Index')
    ax.set_title(f'Pairwise Coincidence Matrix ($C_{{global}}$ = {C_global:.4f})')
    
    # Set ticks to show train indices
    n_trains = C_matrix.shape[0]
    ax.set_xticks(np.arange(n_trains))
    ax.set_yticks(np.arange(n_trains))
    ax.set_xticklabels(np.arange(1, n_trains + 1))
    ax.set_yticklabels(np.arange(1, n_trains + 1))
    
    plt.tight_layout()
    
    return fig, ax
