import numpy as np


def detect_peaks(array):
    size = len(array)

    if size < 3:
        return np.array([])

    diffs = np.hstack((np.inf, np.diff(array), - np.inf))

    raising = diffs[:-1] > 0

    single_peaks = np.where(raising & (diffs[1:] < 0))[0].astype(np.float64)

    multi_peak_candidates = np.where(raising & (diffs[1:] == 0))[0]
    multi_peaks = []

    for peak in multi_peak_candidates:
        cur_ind = peak

        while diffs[cur_ind + 1] == 0:
            cur_ind += 1

        if diffs[cur_ind + 1] < 0:
            if peak == 0 and cur_ind == size - 1:
                return np.array([])

            multi_peaks.append((peak + cur_ind) / 2)

    if not multi_peaks:
        return single_peaks

    ixs = np.searchsorted(single_peaks, multi_peaks)
    peaks = np.insert(single_peaks, ixs, multi_peaks)

    return peaks


def moving_average(array, averaging_len=5):
    if averaging_len <= 0 or not isinstance(averaging_len, int):
        raise ValueError('\'averaging_len\' should be positive integer')

    window = 2 * averaging_len + 1
    size = array.size

    if size < window:
        return array

    array = np.hstack(
        (0, array[:averaging_len], array, array[size - averaging_len:]),
    )

    cumulative = np.cumsum(array)

    average = cumulative[window:].astype(np.float64)
    average -= cumulative[:- window]
    average /= window

    return average


def get_unfolded_nns(peaks, averaging_len=None):
    spacings = np.diff(peaks)

    if not np.all(spacings > 0):
        raise ValueError('\'peaks\' should be sorted array of unique numbers')

    if averaging_len is None:
        return spacings / spacings.mean()

    average = moving_average(spacings, averaging_len=averaging_len)

    return spacings / average
