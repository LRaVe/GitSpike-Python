# ============================================================
# SPIKE-distance computation with auxiliary boundary spikes
# Author: Maxime BELTOISE
# Date: May 2026
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# ============================================================
# PARAMETERS
# ============================================================

measures = 2      # +1:ISI,+2:SPIKE,+4:RI-SPIKE,+8:SPIKE-Synchro,...
showing = 14      # +1:Spike Trains,+2:Distance,+4:Profile,+8:Matrix
plotting = 12     # +1:Spike Trains,+2:Distance,+4:Profile,+8:Matrix


# ============================================================
# INPUT SPIKE TRAINS
# ============================================================

spikes = []
spikes.append([12, 16, 76, 80])
spikes.append([8, 20, 72, 84])
spikes.append([10, 14, 84, 92])
spikes.append([12, 44, 48, 80])
spikes.append([8, 52, 56, 84])
spikes.append([10, 92])

# global time window
t_min = 0
t_max = 100


# ============================================================
# ADD AUXILIARY SPIKES
# ============================================================

def add_auxiliary_spikes(spikes, t_min, t_max):

    aux_begin = False
    aux_end = False

    spikes = sorted(set(spikes))

    # --------------------------------------------------------
    # beginning
    # --------------------------------------------------------
    if spikes[0] > t_min:

        if len(spikes) >= 2:
            aux = spikes[0] - max(spikes[0] - t_min, spikes[1] - spikes[0])
        else:
            aux = t_min

        spikes = [aux] + spikes
        aux_begin = True

    # --------------------------------------------------------
    # end
    # --------------------------------------------------------
    if spikes[-1] < t_max:

        if len(spikes) >= 2:
            aux = spikes[-1] + max(t_max - spikes[-1], spikes[-1] - spikes[-2])
        else:
            aux = t_max

        spikes = spikes + [aux]
        aux_end = True

    return spikes, aux_begin, aux_end


# ============================================================
# AUXILIARY DELTA MANAGEMENT
# ============================================================

def auxiliary_delta(spike, own_train, other_train, idx, aux_idx):

    delta_std = min(abs(spike - x) for x in other_train)
    delta = delta_std

    # auxiliary at beginning
    if idx == 0 and aux_idx:
        delta = min(abs(own_train[1] - x) for x in other_train)

    # auxiliary at end
    if idx == len(own_train) - 1 and aux_idx:
        delta = min(abs(own_train[-2] - x) for x in other_train)

    return delta


# ============================================================
# SPIKE-distance between TWO spike trains
# ============================================================

def SPIKE_dist_2x2(spikes1, spikes2, t_min, t_max):

    # --------------------------------------------------------
    # add auxiliary spikes
    # --------------------------------------------------------

    spikes1, aux1_begin, aux1_end = add_auxiliary_spikes(spikes1, t_min, t_max)
    spikes2, aux2_begin, aux2_end = add_auxiliary_spikes(spikes2, t_min, t_max)

    # --------------------------------------------------------
    # initialize profile
    # --------------------------------------------------------

    profile = []

    # ========================================================
    # LOOP OVER TRAIN 1
    # ========================================================

    for idx1 in range(len(spikes1)):

        t = spikes1[idx1]

        if spikes2[0] > t:
            idx2 = 0

        elif spikes2[-1] <= t:
            idx2 = len(spikes2) - 2

        else:
            idx2 = max(i for i, x in enumerate(spikes2) if x <= t)

        # ----------------------------------------------------
        # train 2 contribution
        # ----------------------------------------------------

        isi2 = spikes2[idx2 + 1] - spikes2[idx2]

        delta_tp2 = auxiliary_delta( spikes2[idx2], spikes2, spikes1, idx2, aux2_begin)

        delta_tf2 = auxiliary_delta(spikes2[idx2 + 1], spikes2, spikes1, idx2 + 1, aux2_end)

        xp2 = t - spikes2[idx2]
        xf2 = spikes2[idx2 + 1] - t

        S2 = ((delta_tp2 * xf2) + (delta_tf2 * xp2)) / isi2

        # ----------------------------------------------------
        # LEFT VALUE
        # ----------------------------------------------------

        if idx1 > 0:

            isi1 = spikes1[idx1] - spikes1[idx1 - 1]

            S1 = auxiliary_delta(spikes1[idx1], spikes1, spikes2, idx1, aux1_end)

            S = ((S1 * isi2) + (S2 * isi1)) / (2 * (np.mean([isi1, isi2]) ** 2))

            profile.append([t, S])

        # ----------------------------------------------------
        # RIGHT VALUE
        # ----------------------------------------------------

        if idx1 < len(spikes1) - 1:

            isi1 = spikes1[idx1 + 1] - spikes1[idx1]

            S1 = auxiliary_delta(spikes1[idx1], spikes1, spikes2, idx1, aux1_begin)

            S = ((S1 * isi2) + (S2 * isi1)) / (2 * (np.mean([isi1, isi2]) ** 2))

            profile.append([t, S])

    # ========================================================
    # LOOP OVER TRAIN 2
    # ========================================================

    for idx2 in range(len(spikes2)):

        t = spikes2[idx2]

        if spikes1[0] > t:
            idx1 = 0

        elif spikes1[-1] <= t:
            idx1 = len(spikes1) - 2

        else:
            idx1 = max(i for i, x in enumerate(spikes1) if x <= t)

        # ----------------------------------------------------
        # train 1 contribution
        # ----------------------------------------------------

        isi1 = spikes1[idx1 + 1] - spikes1[idx1]

        delta_tp1 = auxiliary_delta(spikes1[idx1], spikes1, spikes2, idx1, aux1_begin)

        delta_tf1 = auxiliary_delta(spikes1[idx1 + 1], spikes1, spikes2, idx1 + 1, aux1_end)

        xp1 = t - spikes1[idx1]
        xf1 = spikes1[idx1 + 1] - t

        S1 = ((delta_tp1 * xf1) + (delta_tf1 * xp1)) / isi1

        # ----------------------------------------------------
        # LEFT VALUE
        # ----------------------------------------------------

        if idx2 > 0:

            isi2 = spikes2[idx2] - spikes2[idx2 - 1]

            S2 = auxiliary_delta(spikes2[idx2], spikes2, spikes1, idx2, aux2_end)

            S = ((S2 * isi1) + (S1 * isi2)) / (2 * (np.mean([isi1, isi2]) ** 2))

            profile.append([t, S])

        # ----------------------------------------------------
        # RIGHT VALUE
        # ----------------------------------------------------

        if idx2 < len(spikes2) - 1:

            isi2 = spikes2[idx2 + 1] - spikes2[idx2]

            S2 = auxiliary_delta(spikes2[idx2], spikes2, spikes1, idx2, aux2_begin)

            S = ((S1 * isi2) + (S2 * isi1)) / (2 * (np.mean([isi1, isi2]) ** 2))

            profile.append([t, S])

    # ========================================================
    # SORT (STABLE)
    # ========================================================

    profile = sorted(profile, key=lambda x: x[0])

    # ========================================================
    # CLAMP TO WINDOW
    # ========================================================

    for i in range(len(profile)):

        # before t_min
        if profile[i][0] < t_min:

            idx = next(j for j in range(len(profile)) if profile[j][0] >= t_min)

            x1, y1 = profile[i]
            x2, y2 = profile[idx]

            y = y1 + ((y2 - y1) / (x2 - x1)) * (t_min - x1)

            profile[i] = [t_min, y]

        # after t_max
        elif profile[i][0] > t_max:

            idx = max(j for j in range(len(profile)) if profile[j][0] <= t_max)

            x1, y1 = profile[idx]
            x2, y2 = profile[i]

            y = y1 + ((y2 - y1) / (x2 - x1)) * (t_max - x1)

            profile[i] = [t_max, y]

    # ========================================================
    # REMOVE EXACT DUPLICATES ONLY
    # ========================================================

    cleaned = []

    for p in profile:
        if p not in cleaned:
            cleaned.append(p)

    profile = cleaned

    # stable sort again
    profile = sorted(profile, key=lambda x: x[0])

    # ========================================================
    # FINAL DISTANCE
    # ========================================================

    t = np.array([x[0] for x in profile])
    S = np.array([x[1] for x in profile])

    D = np.trapz(S, t) / (t_max - t_min)

    return D, profile


# ============================================================
# SPIKE-distance for N spike trains
# ============================================================

def SPIKE_dist_N(spikes, t_min, t_max):

    N = len(spikes)

    D_matrix = np.zeros((N, N))

    profiles = []

    # ========================================================
    # pairwise distances
    # ========================================================

    for i, j in combinations(range(N), 2):

        d, prof = SPIKE_dist_2x2(spikes[i], spikes[j], t_min, t_max)

        D_matrix[i, j] = d
        D_matrix[j, i] = d

        profiles.append(prof)

    # ========================================================
    # global distance
    # ========================================================

    D_global = np.mean(D_matrix[np.triu_indices(N, 1)])

    # ========================================================
    # all time coordinates
    # ========================================================

    t_all = []

    for P in profiles:
        for pt in P:
            t_all.append(pt[0])

    t_all = sorted(set(t_all))

    # ========================================================
    # global profile
    # ========================================================

    profile_global = []

    for t in t_all:

        vals_left = []
        vals_right = []

        has_discontinuity = False

        # ----------------------------------------------------
        # scan all pairwise profiles
        # ----------------------------------------------------

        for P in profiles:

            idx = [i for i, pt in enumerate(P) if np.isclose(pt[0], t)]

            # =================================================
            # CASE 1 : discontinuity
            # =================================================

            if len(idx) == 2:
                has_discontinuity = True

                vals_left.append(P[idx[0]][1])
                vals_right.append(P[idx[1]][1])

            # =================================================
            # CASE 2 : single point
            # =================================================

            elif len(idx) == 1:
                vals_left.append(P[idx[0]][1])
                vals_right.append(P[idx[0]][1])

            # =================================================
            # CASE 3 : interpolation
            # =================================================

            else:
                before = [i for i, pt in enumerate(P) if pt[0] < t]
                after = [i for i, pt in enumerate(P) if pt[0] > t]

                if len(before) > 0 and len(after) > 0:

                    i1 = before[-1]
                    i2 = after[0]

                    t1, S1 = P[i1]
                    t2, S2 = P[i2]

                    if t2 != t1:
                        S_interp = S1 + ((S2 - S1) * (t - t1) / (t2 - t1))
                    else:
                        S_interp = S1

                    vals_left.append(S_interp)
                    vals_right.append(S_interp)

        # ----------------------------------------------------
        # averaging
        # ----------------------------------------------------

        if has_discontinuity:

            profile_global.append([t, np.mean(vals_left)])
            profile_global.append([t, np.mean(vals_right)])

        else:

            profile_global.append([t, np.mean(vals_left)])

    # stable sort
    profile_global = sorted(profile_global, key=lambda x: x[0])

    return D_global, profile_global, D_matrix





# ============================================================
# DISPLAY
# ============================================================

if measures % 4 > 1:                                                            #SPIKE-distance
    D_global, profile_global, D_matrix = SPIKE_dist_N(spikes, t_min, t_max)
    if showing % 4 > 1:
        print("\nSPIKE-distance:\n")
        print(D_global)
    if showing % 8 > 3:
        print("\nSPIKE profile:\n")
        for p in profile_global:
            print(p)
    if showing % 16 > 7:
        print("\nDistance matrix:\n")
        print(D_matrix)
    if plotting % 8 > 3:
        profile_arr = np.array(profile_global)
        plt.figure()
        plt.fill_between(profile_arr[:, 0], profile_arr[:, 1], alpha=0.5)
        plt.plot(profile_arr[:, 0], profile_arr[:, 1])
        plt.xlabel("Time")
        plt.ylabel("SPIKE distance")
        plt.title(f"SPIKE-distance = {D_global}")
        plt.xlim([t_min, t_max])
        plt.ylim([0, 1])
        plt.grid(True)

    if plotting % 16 > 7:
        plt.figure()
        plt.imshow(D_matrix, cmap="jet", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Spike trains")
        plt.ylabel("Spike trains")
        plt.title(f"SPIKE-distance = {D_global}")

plt.show()