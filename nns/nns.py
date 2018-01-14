import numpy as np


def get_peaks(array):
    peaks = []

    start = -1
    cur_val = - np.inf

    for ix, next_val in enumerate(array):
        if next_val > cur_val:
            start = ix

        elif start != -1 and next_val != cur_val:
            peaks.append((start + ix - 1) / 2)
            start = -1

        cur_val = next_val

    if start != -1:
        peaks.append((start + ix - 1) / 2)

    return peaks


def get_unfolded_nns(peaks, window=5):
    spacings = np.diff(peaks)

    length = len(spacings)
    unfolded_spacings = np.zeros(length)

    for ix in range(length):
        start = max(0, ix - window)
        end = min(length, ix + window + 1)

        unfolded_spacings[ix] = spacings[ix] / spacings[start:end].mean()

    return unfolded_spacings
