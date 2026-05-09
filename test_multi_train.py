"""
Test multi-train spike synchronization with corrected algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from f_spike_synchro_multi import f_spike_synchro_multi
from f_spike_synchro_plot import f_spike_synchro_plot

# Choose dataset
dataset = 1

if dataset == 1:
    t_min = 0
    t_max = 10
    train1 = np.array([0, 1.9, 3.9, 7, 10])
    train2 = np.array([0, 2, 7.1, 9, 10])
    train3 = np.array([0, 2.1, 4.1, 6.9, 10])
    spikes = [train1, train2, train3]
    
    print('Train 1:', train1)
    print('Train 2:', train2)
    print('Train 3:', train3)
    print()

elif dataset == 2:
    t_min = 0
    t_max = 100
    spikes = [
        np.array([12, 16, 28, 32, 44, 48, 60, 64, 76, 80]),
        np.array([8, 20, 24, 36, 40, 52, 56, 68, 72, 84])
    ]
    print('Train 1:', spikes[0])
    print('Train 2:', spikes[1])
    print()

else:
    raise ValueError('Invalid dataset selection. Please choose 1 or 2.')


# Call multi-train function
C_matrix, C_global = f_spike_synchro_multi(spikes, t_min, t_max)

print('=== Pairwise Coincidence Matrix ===')
print(C_matrix)

print('\n=== Global SPIKE-Synchronization Index ===')
print(f'C_global: {C_global:.4f}')

# Plot results
print('\nGenerating plot...')
fig, ax = f_spike_synchro_plot(C_matrix, C_global)
plt.show()
