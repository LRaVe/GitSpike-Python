# Module for plotting figures for the discriminative subpopulation analysis.
# Based on original MATLAB code (Maxime Beltoise), translated for PySpike.
# Distributed under the BSD License

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from pyspike.algo.pooling import pool_neurons
from pyspike.algo.discrimination import compute_discrimination_performance
from pyspike.algo.labeled_line import evaluate_LL_population  # if needed elsewhere

############################################################
# plot_SP_figure
############################################################
def plot_SP_figure(spikes, params, plot_params):
    """ Reproduces Fig. 1 of Satuvuori et al. (2018): raster plots of
    individual spike trains for selected stimulus/repetition pairs,
    optionally with the pooled coding ('C'), non-coding ('NC') and
    full population ('All') spike trains shown below.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param params: dict, same as used in :func:`.generate_SP_dataset`
                   (keys 'c', 'nIndi', 'Tmax' are used here).
    :param plot_params: dict with the keys:

        - stimuli: iterable of stimulus indices to show (0-indexed).
        - repetitions: iterable of repetition indices to show
          (0-indexed).
        - showPooling: bool, whether to display the pooled C/NC/All
          spike trains below the individual rasters.
    """
    N = spikes.shape[0]
    c = params['c']
    n_indi = params['nIndi']
    Tmax = params['Tmax']

    coding_neurons = range(0, c)
    indi_neurons = range(c, c + n_indi)
    # NB: as in the original MATLAB code, the pooled 'NC' group actually
    # combines the Indi AND the non-coding neurons (everything after the
    # coding block), it is not restricted to the true non-coding neurons.
    non_coding = range(c, N)

    stimuli = plot_params['stimuli']
    repetitions = plot_params['repetitions']
    show_pooling = plot_params['showPooling']

    n_panels = len(stimuli) * len(repetitions)

    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels))
    fig.patch.set_facecolor('w')
    if n_panels == 1:
        axes = [axes]

    plot_index = 0

    for s in stimuli:
        for r in repetitions:

            ax = axes[plot_index]

            # =====================================
            # AXES
            # =====================================
            ax.set_xlim(0, Tmax)

            # =====================================
            # Y POSITIONS
            # =====================================
            # individual neurons: neuron 0 at the top (y=N), neuron N-1 at
            # the bottom (y=1), matching the MATLAB neuronY = N:-1:1
            neuron_y = [N - n for n in range(N)]

            y_c, y_nc, y_all = 0, -1, -2

            # =====================================
            # INDIVIDUAL SPIKE TRAINS
            # =====================================
            for n in range(N):

                t = spikes[n, s, r].spikes

                if n in coding_neurons:
                    col = (1, 0, 0)
                elif n in indi_neurons:
                    col = (1, 0, 1)
                else:
                    col = (0, 0, 1)

                y = neuron_y[n]

                ax.vlines(t, y - 0.4, y + 0.4, color=col, linewidth=1.5)

            # =====================================
            # DASHED SEPARATION LINES
            # coding / indi / non-coding
            # =====================================
            separation_y = N - c + 0.5
            ax.axhline(separation_y, linestyle='--', color=(0.4, 0.4, 0.4),
                      linewidth=1.2)

            separation_y2 = N - (c + n_indi) + 0.5
            ax.axhline(separation_y2, linestyle='--', color=(0.4, 0.4, 0.4),
                      linewidth=1.2)

            # =====================================
            # POOLING
            # =====================================
            if show_pooling:

                # -----------------
                # C
                # -----------------
                pooled_c = pool_neurons(spikes, coding_neurons, s, r)
                ax.vlines(pooled_c, y_c - 0.4, y_c + 0.4, color=(1, 0, 0),
                         linewidth=1.5)
                ax.axhline(y_c + 0.5, color='k', linewidth=1)

                # -----------------
                # NC
                # -----------------
                pooled_nc = pool_neurons(spikes, non_coding, s, r)
                ax.vlines(pooled_nc, y_nc - 0.4, y_nc + 0.4, color=(0, 0, 1),
                         linewidth=1.5)
                ax.axhline(y_nc + 0.5, linestyle='--', color=(0.4, 0.4, 0.4),
                          linewidth=1.2)

                # -----------------
                # ALL
                # -----------------
                pooled_all = pool_neurons(spikes, range(N), s, r)
                ax.vlines(pooled_all, y_all - 0.4, y_all + 0.4, color='k',
                         linewidth=1.5)
                ax.axhline(y_all + 0.5, linestyle='--', color=(0.4, 0.4, 0.4),
                          linewidth=1.2)

                # =====================================
                # TICKS
                # =====================================
                ax.set_yticks([y_all, y_nc, y_c,
                              neuron_y[-1], neuron_y[c], neuron_y[0]])
                ax.set_yticklabels(['All', 'NC', 'C',
                                   f'N{N}', f'N{c + 1}', 'N1'])

            else:
                ax.set_yticks([neuron_y[0], neuron_y[-1]])
                ax.set_yticklabels(['1', str(N)])

            # =====================================
            # remove tick marks
            # =====================================
            ax.tick_params(length=0)

            # =====================================
            # LIMITS
            # =====================================
            ax.set_ylim(y_all - 1, N + 1)

            # =====================================
            # STYLE
            # =====================================
            ax.set_xlabel('Time')
            ax.set_ylabel('Spike trains')
            ax.set_title(f'S{s + 1}-R{r + 1}', fontweight='bold')
            ax.tick_params(labelsize=11)
            for spine in ax.spines.values():
                spine.set_visible(True)

            plot_index += 1

    fig.tight_layout()
    return fig


############################################################
# plot_distance_matrix
############################################################
def plot_distance_matrix(D, labels, title_str, ax=None):
    """ Reproduces Fig. 2 of Satuvuori et al. (2018): a color-coded
    pairwise distance matrix with the discrimination performance P
    in the title, and separation lines between stimulus blocks.

    :param D: symmetric (T, T) distance matrix.
    :param labels: `np.ndarray` of shape (T,) with the stimulus index
                   of each trial (as returned by :func:`.build_trials`).
    :param title_str: label prefixed to the discrimination performance
                      in the title (e.g. 'C', 'NC', 'All').
    :param ax: optional matplotlib Axes to draw on; a new figure/axes
              is created if not given.
    :returns: (fig, ax)
    """
    P = compute_discrimination_performance(D, labels)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(D, cmap='jet', aspect='equal')
    fig.colorbar(im, ax=ax)

    ax.set_title(f'{title_str}   |   P = {P:.3f}', fontweight='bold')
    ax.set_xlabel('Recording')
    ax.set_ylabel('Recording')

    labels = np.asarray(labels)
    S = labels.max() + 1
    R = np.sum(labels == 0)
    T = len(labels)

    trial_labels = [f'S{s + 1}R{r + 1}' for s in range(S) for r in range(R)]

    max_labels = 10
    if T <= max_labels:
        tick_pos = np.arange(T)
        tick_labels = trial_labels
    else:
        tick_pos = np.unique(
            np.round(np.linspace(0, T - 1, max_labels)).astype(int))
        tick_labels = [trial_labels[i] for i in tick_pos]

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(length=0)

    for s in range(1, S):
        pos = s * R - 0.5
        ax.axvline(pos, color='k', linewidth=2)
        ax.axhline(pos, color='k', linewidth=2)

    return fig, ax


############################################################
# plot_brute_force
############################################################
def plot_brute_force(result):
    """ Plots the brute force search history: performance of every
    evaluated subpopulation (enumeration order), the running
    best-so-far, and the global maximum.

    :param result: dict as returned by :func:`.brute_force_search`.
    :returns: matplotlib Figure.
    """
    history = result['historyPerf']
    total_combinations = len(history)
    best_perf_overall = result['bestPerformance']

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('w')

    x = np.arange(1, total_combinations + 1)
    ax.plot(x, history, color=(0.5, 0.5, 0.5), linewidth=0.8,
           label='Evaluated Mask Performance')

    best_so_far = np.maximum.accumulate(history)
    ax.plot(x, best_so_far, 'b-', linewidth=2, label='Global Maximum Progress')

    idx_max = int(np.argmax(history == best_perf_overall))
    ax.plot(x[idx_max], best_perf_overall, 'o', color=(1, 0.2, 0.2),
           markersize=8, label='Absolute Best Solution')

    ax.grid(True)
    ax.set_xlim(1, total_combinations)

    min_p = max(0, np.min(history))
    max_p = np.max(history)
    ax.set_ylim(min_p, max(max_p * 1.1, 0.1))

    ax.set_xlabel('Binary Counter Iterations (Search Space)', fontweight='bold')
    ax.set_ylabel('Performance P', fontweight='bold')
    N = int(np.log2(total_combinations + 1))
    ax.set_title(f'Brute Force Search Tree Exploration (N = {N} Neurons)',
               fontweight='bold')
    ax.legend(loc='lower right')

    return fig


############################################################
# plot_bottom_up
############################################################
def plot_bottom_up(result, other_figs=True):
    """ Plots the Bottom-Up algorithm results: (optionally) the
    performance-vs-step overview and the raw selection matrix, and
    (always) the "paper-style" combined matrix + performance summary
    figure (analogous to Fig. 3A of Satuvuori et al., 2018).

    :param result: dict as returned by :func:`.bottom_up_search`.
    :param other_figs: if True, also draws the overview line plot and
                       the raw selection matrix.
    :returns: list of matplotlib Figures created.
    """
    best_order = result['bestOrder']
    history_perf = result['historyPerf']
    matrix_grid = result['matrixGrid']
    best_subpop = result['bestSubpopulation']
    num_neurons = len(best_order)

    figs = []

    if other_figs:

        # 1. Overview plot
        idx_max = int(np.argmax(history_perf))
        max_p = history_perf[idx_max]

        fig1, ax1 = plt.subplots()
        steps = np.arange(1, num_neurons + 1)
        ax1.plot(steps, history_perf, '-o', linewidth=2.5,
                color=(0.30, 0.58, 0.20), markerfacecolor=(0.93, 0.69, 0.13),
                markeredgecolor=(0.30, 0.58, 0.20), markersize=8)
        ax1.axvline(idx_max + 1, color=(0.85, 0.33, 0.1), linestyle='--',
                   linewidth=1.5)
        ax1.grid(True)
        ax1.set_xlim(0.5, num_neurons + 0.5)
        ax1.set_ylim(history_perf.min() - 0.02, history_perf.max() + 0.04)
        ax1.set_xlabel('Neurons integrated sequentially (Step k)', fontweight='bold')
        ax1.set_ylabel('Global performance P', fontweight='bold')
        ax1.set_title('Evolution of performance using Bottom-Up selection',
                     fontweight='bold')
        ax1.annotate(
            f'Optimal subpopulation:\nNeurons: {best_subpop}\nMax P = {max_p:.4f}',
            xy=(idx_max + 1, max_p), fontsize=9, fontweight='bold',
            bbox=dict(facecolor=(0.96, 0.96, 0.96), edgecolor=(0.7, 0.7, 0.7)))
        fig1.tight_layout()
        figs.append(fig1)

        # 2. Raw selection matrix
        fig2, ax2 = plt.subplots()
        masked = np.ma.masked_invalid(matrix_grid)
        im = ax2.imshow(masked, cmap='jet', aspect='auto')
        fig2.colorbar(im, ax=ax2)

        for step, neuron in enumerate(best_order):
            ax2.plot(neuron, step, 'kx', markersize=12, linewidth=2.5)

        min_p2, max_p2 = history_perf.min(), history_perf.max()
        scaled_perf = (history_perf - min_p2) / (max_p2 - min_p2) * (num_neurons - 1)
        ax2.plot(scaled_perf, np.arange(num_neurons), '-r', linewidth=2.5)

        ax2.set_xlabel('# Neuron', fontweight='bold')
        ax2.set_ylabel('Number of neurons (Step k)', fontweight='bold')
        ax2.set_title('Bottom-Up selection matrix', fontweight='bold')
        fig2.tight_layout()
        figs.append(fig2)

    # 3. Paper-style figure
    opt_size = len(best_subpop)
    min_perf_val = history_perf.min() - 0.02

    matrix_paper = matrix_grid.copy()
    for k in range(num_neurons):
        past_neurons = best_order[:k]
        matrix_paper[k, past_neurons] = min_perf_val
        matrix_paper[k, best_order[k]] = history_perf[k]

    fig3 = plt.figure(figsize=(12, 5))
    gs = fig3.add_gridspec(1, 5)
    ax_left = fig3.add_subplot(gs[0, 0:3])
    ax_right = fig3.add_subplot(gs[0, 3:5])

    im = ax_left.imshow(matrix_paper, cmap='jet', origin='lower',
                       vmin=min_perf_val, vmax=history_perf.max() + 0.02,
                       aspect='auto')

    for k in range(num_neurons):
        n_id = best_order[k]
        ax_left.text(n_id, k, '\u2713', fontsize=11, ha='center', va='center',
                    fontweight='bold', color='k')

    for k_sub in range(opt_size):
        curr_n = best_order[k_sub]
        ax_left.plot(curr_n, opt_size - 1, 'rx', markersize=12, linewidth=2)

    ax_left.set_xlabel('Neuron ID', fontweight='bold')
    ax_left.set_ylabel('Size of population (k)', fontweight='bold')
    ax_left.set_title('Bottom-Up algorithm matrix', fontweight='bold')
    fig3.colorbar(im, ax=ax_left, label='Global Performance P')

    ax_right.plot(history_perf, np.arange(1, num_neurons + 1), '-ko',
                 linewidth=2, markerfacecolor='k', markersize=5)
    ax_right.plot(history_perf[opt_size - 1], opt_size, 'o', color='r',
                 markersize=11, linewidth=2, markerfacecolor='none')
    ax_right.plot(history_perf[opt_size - 1], opt_size, 'rx', markersize=7,
                 linewidth=1.5)
    ax_right.grid(True)
    ax_right.set_ylim(0.5, num_neurons + 0.5)
    ax_right.set_xlabel('Best performance P', fontweight='bold')
    ax_right.set_title('Performance function', fontweight='bold')

    fig3.tight_layout()
    figs.append(fig3)

    return figs


############################################################
# plot_top_down
############################################################
def plot_top_down(result, coding_neurons):
    """ Reproduces Fig. 3B of Satuvuori et al. (2018): a heatmap of the
    discrimination performance of every candidate subpopulation
    evaluated by the top-down gradient search, plus the best
    performance reached at each population size.

    :param result: dict as returned by :func:`.top_down`.
    :param coding_neurons: iterable of the ground-truth coding neuron
                           indices (0-indexed), used only to draw the
                           coding/non-coding separation line.
    :returns: matplotlib Figure.
    """
    N = len(result['populations'][0])

    M = np.full((N, N), np.nan)
    M[N - 1, 0] = result['P'][0]

    for step, candidate_p in enumerate(result['candidateP']):
        pop_size = N - (step + 1)
        M[pop_size - 1, :] = candidate_p

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    im = ax_left.imshow(M, origin='lower', cmap='jet', aspect='auto')
    ax_left.tick_params(length=0)
    fig.colorbar(im, ax=ax_left)
    ax_left.set_title('Top-Down', fontweight='bold')
    ax_left.set_xlabel('Neuron Index')
    ax_left.set_ylabel('Size of population')

    for step, neuron in enumerate(result['removedNeuron']):
        pop_size = N - (step + 1)
        ax_left.plot(neuron, pop_size - 1, marker='_', color='k',
                    markersize=10, linewidth=2)

    ax_left.plot(0, N - 1, marker='o', color='k', markerfacecolor='k',
                markersize=5, linewidth=2)

    best_size = len(result['bestPopulation'])
    for n in result['bestPopulation']:
        ax_left.plot(n, best_size - 1, marker='x', color='m',
                    markersize=12, linewidth=2)

    ax_left.axvline(len(list(coding_neurons)) - 0.5, color='k', linewidth=4)

    pop_sizes = [len(p) for p in result['populations']]
    ax_right.plot(result['P'], pop_sizes, '-ok', linewidth=2,
                 markerfacecolor='k')
    ax_right.plot(result['bestP'], best_size, marker='o', color='m',
                 markersize=14, linewidth=3, markerfacecolor='none')
    ax_right.plot(result['bestP'], best_size, marker='x', color='m',
                 markersize=18, linewidth=3)
    ax_right.set_xlabel('Best performance per pop size')
    ax_right.set_ylim(0.5, N + 0.5)
    ax_right.grid(True)
    ax_right.tick_params(length=0)

    fig.tight_layout()
    return fig


############################################################
# plot_simulated_annealing
############################################################
def plot_simulated_annealing(result, coding_neurons):
    """ Reproduces the simulated annealing diagnostic figure: evolution
    of population size, discrimination performance, population
    composition, cooling schedule, and subpopulation purity over the
    search.

    :param result: dict as returned by :func:`.simulated_annealing`.
    :param coding_neurons: iterable of ground-truth coding neuron
                           indices (0-indexed), used only to draw
                           reference lines.
    :returns: matplotlib Figure.
    """
    history = result['history']
    n_coding_gt = len(list(coding_neurons))

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    fig.patch.set_facecolor('w')

    iterations = np.arange(len(history['P']))

    # Population size
    ax = axes[0, 0]
    ax.plot(iterations, history['size'], 'b', linewidth=1.5)
    ax.axhline(n_coding_gt, color='r', linestyle='--', label='Ground truth')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Population size')
    ax.set_title('Population size')
    ax.legend(loc='best')
    ax.grid(True)

    # Performance
    ax = axes[0, 1]
    ax.plot(iterations, history['P'], 'k', linewidth=1.5, label='Current')
    ax.plot(iterations, history['bestP'], 'r', linewidth=2, label='Best')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('P')
    ax.set_title('Discrimination performance')
    ax.legend(loc='best')
    ax.grid(True)

    # Coding / non-coding composition
    ax = axes[1, 0]
    ax.plot(iterations, history['nCoding'], 'g', linewidth=1.5, label='Coding')
    ax.plot(iterations, history['nNonCoding'], 'r', linewidth=1.5, label='Non-coding')
    ax.axhline(n_coding_gt, color='g', linestyle='--', label='All coding neurons')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Neuron count')
    ax.set_title('Population composition')
    ax.legend(loc='best')
    ax.grid(True)

    # Temperature
    ax = axes[1, 1]
    ax.semilogy(iterations, history['temperature'], 'm', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Temperature')
    ax.set_title('Cooling schedule')
    ax.grid(True)

    # Purity
    ax = axes[2, 0]
    purity = history['nCoding'] / history['size']
    ax.plot(iterations, 100 * purity, 'k', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Purity')
    ax.set_title('Population composition')
    ax.grid(True)

    axes[2, 1].axis('off')

    best_pop_str = ' '.join(str(n) for n in result['bestPopulation'])
    fig.suptitle(
        f"Simulated Annealing | Best P = {result['bestP']:.3f} | "
        f"Best Population = [{best_pop_str}]")

    fig.tight_layout()
    return fig


############################################################
# draw_separators
############################################################
def draw_separators(ax, n_stimuli, n_repeats, linewidth, color='k'):
    """ Draws grid separator lines on `ax`, dividing an
    (n_stimuli * n_repeats) x (n_stimuli * n_repeats) matrix plot into
    blocks of `n_repeats` trials per stimulus. With n_repeats=1 (as
    used throughout the LL plotting functions), this draws a full
    grid over an (n_stimuli, n_stimuli) matrix.

    :param ax: matplotlib Axes to draw on.
    :param n_stimuli: number of stimuli (S).
    :param n_repeats: block size (1 for S x S stimulus-level matrices,
                      R for T x T trial-level matrices).
    :param linewidth: line width of the separators.
    :param color: line color (default black, as in the original).
    """
    total_extent = n_stimuli * n_repeats
    separators = np.arange(1, n_stimuli) * n_repeats

    for sep in separators:
        # convert from MATLAB's 1-indexed pixel grid to Python's
        # 0-indexed imshow pixel grid (same -0.5 shift as in
        # plot_distance_matrix)
        pos = sep - 0.5
        ax.plot([-0.5, total_extent - 0.5], [pos, pos],
               color=color, linewidth=linewidth)
        ax.plot([pos, pos], [-0.5, total_extent - 0.5],
               color=color, linewidth=linewidth)



def _black_prefixed_continuous_cmap(base='jet', n=256):
    """ Approximates MATLAB's `colormap([0,0,0; jet])`. """
    base_colors = plt.get_cmap(base, n)(np.linspace(0, 1, n))
    colors = np.vstack([[0, 0, 0, 1], base_colors])
    return ListedColormap(colors)


def _neuron_categorical_cmap(N, hue_max=0.85):
    """ Approximates MATLAB's `colormap([0,0,0; hsv(N)])`: black for
    'no neuron' (value 0), N distinct hues for neurons 1..N
    (1-indexed for display only).

    Sampled over [0, hue_max] rather than the full [0, 1] hue circle,
    so the last neuron lands on pink/violet (~hue 0.85) instead of
    wrapping back around to red (hue 0 == hue 1 on the HSV wheel,
    which made the first and last neurons look almost identical).
    """
    hsv_colors = plt.get_cmap('hsv')(np.linspace(0, hue_max, N))
    colors = np.vstack([[0, 0, 0, 1], hsv_colors])
    return ListedColormap(colors)


############################################################
# plot_LL_results
############################################################
def plot_LL_results(spikes, result):
    """ Diagnostic figures for the Labeled Line (LL) analysis: spike
    raster grid, per-neuron pairwise distance matrices, per-neuron
    discrimination and performance matrices, the population
    performance matrix, and the best-neuron matrix.

    :param spikes: numpy object array of shape (N, S, R) of
                   :class:`.SpikeTrain`.
    :param result: dict as returned by :func:`.evaluate_LL_population`.
    :returns: list of the matplotlib Figures created.
    """
    N, S, R = spikes.shape
    figs = []
    n_cols = int(np.ceil(N / 2))

    cmap_d = _black_prefixed_continuous_cmap('jet')
    cmap_neuron = _neuron_categorical_cmap(N)
    norm_neuron = BoundaryNorm(np.arange(-0.5, N + 1.5, 1), cmap_neuron.N)

    # =====================================================
    # 1. Structured spike matrix
    # =====================================================
    max_t = 0.0
    for n in range(N):
        for s in range(S):
            for r in range(R):
                t = spikes[n, s, r].spikes
                if len(t) > 0:
                    max_t = max(max_t, t.max())

    fig1, axes1 = plt.subplots(N, S, figsize=(2 * S, 1.5 * N), squeeze=False)
    fig1.patch.set_facecolor('w')

    for n in range(N):
        for s in range(S):
            ax = axes1[n, s]
            for r in range(R):
                st = spikes[n, s, r].spikes
                if len(st) == 0:
                    continue
                ax.plot(st, (r + 1) * np.ones_like(st), 'k|', markersize=4)

            ax.set_xlim(0, max_t)
            ax.set_ylim(0, R + 1)
            if n == 0:
                ax.set_title(f'Stimuli{s + 1}')
            if n == N - 1:
                ax.set_xlabel('Time')
            if s == 0:
                ax.set_ylabel(f'Neuron{n + 1}')
            ax.set_xticks([])
            ax.set_yticks([])

    fig1.suptitle('Spike trains Matrix')
    fig1.tight_layout()
    figs.append(fig1)

    # =====================================================
    # 2. Pairwise distance matrices D_n
    # =====================================================
    fig2, axes2 = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    fig2.patch.set_facecolor('w')
    axes2 = axes2.ravel()

    for n in range(N):
        ax = axes2[n]
        im = ax.imshow(result['DistanceMatrix'][n], cmap=cmap_d)
        ax.set_title(f'Neuron {n + 1}')
        ax.set_xlabel('Trials')
        ax.set_ylabel('Trials')
        fig2.colorbar(im, ax=ax)

    for ax in axes2[N:]:
        ax.axis('off')

    fig2.suptitle('Pairwise SPIKE distance matrices D_n')
    fig2.tight_layout()
    figs.append(fig2)

    # =====================================================
    # 3. Discrimination matrices
    # =====================================================
    fig3, axes3 = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    fig3.patch.set_facecolor('w')
    axes3 = axes3.ravel()

    for n in range(N):
        ax = axes3[n]
        # +1: convert the 0-indexed neuron id to a 1-indexed display
        # value, consistent with the bestNeuronMatrix sentinel fix
        ax.imshow(result['Discrimination'][n] * (n + 1), cmap=cmap_neuron,
                 norm=norm_neuron)
        ax.set_xticks(range(S))
        ax.set_xticklabels(range(1, S + 1))
        ax.set_yticks(range(S))
        ax.set_yticklabels(range(1, S + 1))
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('Stimulus')
        ax.set_title(f'M_{{{n + 1}}}')
        draw_separators(ax, S, 1, 1.5)

    for ax in axes3[N:]:
        ax.axis('off')

    fig3.suptitle('Discrimination matrices')
    fig3.tight_layout()
    figs.append(fig3)

    # =====================================================
    # 4. Performance matrices Mn
    # =====================================================
    max_performance = max(mn.max() for mn in result['Mn'])

    fig4, axes4 = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    fig4.patch.set_facecolor('w')
    axes4 = axes4.ravel()

    for n in range(N):
        ax = axes4[n]
        im = ax.imshow(result['Mn'][n], cmap=cmap_d, vmin=0, vmax=max_performance)
        ax.set_title(f'P_{{{n + 1}}}')
        fig4.colorbar(im, ax=ax)
        draw_separators(ax, S, 1, 1.5)

    for ax in axes4[N:]:
        ax.axis('off')

    fig4.suptitle('Performance matrices')
    fig4.tight_layout()
    figs.append(fig4)

    # =====================================================
    # 5. Population performance matrix
    # =====================================================
    fig5, ax5 = plt.subplots()
    fig5.patch.set_facecolor('w')
    im = ax5.imshow(result['populationPerformance'], cmap=cmap_d)
    fig5.colorbar(im, ax=ax5)
    ax5.set_title(f"Population performance (PLL = {result['bestP']:.3f})",
                 fontweight='bold')
    ax5.set_xlabel('Stimulus')
    ax5.set_ylabel('Stimulus')
    ax5.set_xticks(range(S))
    ax5.set_xticklabels(range(1, S + 1))
    ax5.set_yticks(range(S))
    ax5.set_yticklabels(range(1, S + 1))
    draw_separators(ax5, S, 1, 1.5)
    figs.append(fig5)

    # =====================================================
    # 6. Best neuron matrix
    # =====================================================
    display_best_neuron = np.where(result['bestNeuronMatrix'] >= 0,
                                  result['bestNeuronMatrix'] + 1, 0)

    fig6, ax6 = plt.subplots()
    fig6.patch.set_facecolor('w')
    ax6.imshow(display_best_neuron, cmap=cmap_neuron, norm=norm_neuron)
    ax6.set_xlabel('Stimulus')
    ax6.set_ylabel('Stimulus')
    ax6.set_xticks(range(S))
    ax6.set_xticklabels(range(1, S + 1))
    ax6.set_yticks(range(S))
    ax6.set_yticklabels(range(1, S + 1))
    best_pop_str = ' '.join(str(n+1) for n in result['bestPopulation'])
    ax6.set_title(f'Best neurons  [{best_pop_str}]', fontweight='bold')
    draw_separators(ax6, S, 1, 1.5)
    figs.append(fig6)

    print()
    print('=========== LL RESULTS ===========')
    print('Best population :', ' '.join(str(n+1) for n in result['bestPopulation']))
    print(f"Performance PLL : {result['bestP']:.4f}")
    print('===================================')

    return figs


############################################################
# plot_complexity_benchmark
############################################################
def plot_complexity_benchmark(result):
    """ Reproduces Fig. 8 of Satuvuori et al. (2018): log-scale
    comparison of the number of subpopulations evaluated by each SP
    search strategy as a function of the number of neurons N.

    :param result: dict as returned by :func:`.run_complexity_benchmark`.
    :returns: matplotlib Figure.
    """
    neuron_counts = result['neuronCounts']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('w')

    ax.plot(neuron_counts, result['greedy'], '-o', linewidth=2, markersize=6,
           color=(0, 0.4470, 0.7410), markerfacecolor=(0, 0.4470, 0.7410),
           label='Bottom-Up/Top-Down (Polynomial: N(N+1)/2)')

    ax.plot(neuron_counts, result['saIterations'], '-s', linewidth=2,
           markersize=6, color=(0.8500, 0.3250, 0.0980),
           markerfacecolor=(0.8500, 0.3250, 0.0980),
           label='Simulated Annealing (Heuristic)')

    ax.plot(neuron_counts, result['saUnique'], '-^', linewidth=2, markersize=6,
           color='r', markerfacecolor='r',
           label='Simulated Annealing (Heuristic and unique)')

    ax.plot(neuron_counts, result['bruteForce'], '-D', linewidth=2,
           markersize=6, color=(0.4940, 0.1840, 0.5560),
           markerfacecolor=(0.4940, 0.1840, 0.5560),
           label='Brute Force (Exponential: 2^N-1)')

    ax.grid(True)
    ax.set_xticks(neuron_counts)
    ax.set_xlabel('Number of Neurons in Pool (N)', fontweight='bold')
    ax.set_ylabel('Number of Evaluated Subpopulations', fontweight='bold')
    ax.set_title('Search Space Exploration Scale (Log Scale)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    fig.tight_layout()
    return fig