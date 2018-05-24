import numpy as np


def detect_peaks(array):
    size = array.size

    if size < 3:
        return np.array([])

    diff = np.hstack((np.inf, np.diff(array), - np.inf))

    _ind = np.where(diff != 0)[0]

    if np.array_equal(_ind, [0, size]):
        return np.array([])

    _peaks = np.where((diff[_ind][:-1] > 0) & (diff[_ind][1:] < 0))[0]

    peaks = (_ind[:-1][_peaks] + _ind[1:][_peaks] - 1) / 2

    return peaks


def detect_significant_peaks(array, threshold=0, min_height=None):
    size = array.size

    if size < 3:
        return np.array([])

    diff = np.hstack((np.inf, np.diff(array), - np.inf))

    peaks = np.where((diff[:-1] > threshold) & (diff[1:] < - threshold))[0]

    if min_height is not None:
        peaks = peaks[array[peaks] >= min_height]

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


def unfolded_nns(peaks, averaging_len=None):
    spacings = np.diff(peaks)

    if not np.all(spacings > 0):
        raise ValueError('\'peaks\' should be sorted array of unique numbers')

    if averaging_len is None:
        return spacings / spacings.mean()

    average = moving_average(spacings, averaging_len=averaging_len)

    return spacings / average
