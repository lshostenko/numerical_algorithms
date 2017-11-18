import numpy as np


def get_peaks_bounds(array):
    peaks = []

    lenght = len(array)
    cur_val = 0
    cur_candidate = []

    for ix in range(lenght):
        next_val = array[ix]

        if next_val > cur_val:
            cur_candidate = [ix]

        elif cur_candidate:
            if next_val == cur_val:
                cur_candidate.append(ix)

            else:
                peaks.append(cur_candidate)
                cur_candidate = []

        cur_val = next_val

    if cur_candidate:
        peaks.append(cur_candidate)

    return peaks


def get_peaks(array):
    peaks_bounds = get_peaks_bounds(array)
    peaks = [sum(bound) / len(bound) for bound in peaks_bounds]

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
