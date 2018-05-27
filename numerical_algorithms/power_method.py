import numpy as np
from scipy.sparse.csgraph import connected_components


def _is_markov_matrix(matrix):
    shape = matrix.shape

    is_markov = len(shape) == 2 and \
        shape[0] == shape[1] and \
        np.allclose(matrix.sum(axis=1), 1)

    return is_markov


def _power_method(transition_matrix, increase_power=True):
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    while True:
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)


def connected_nodes(matrix):
    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups


def stationary_distribution(
    transition_matrix,
    increase_power=True,
    normalized=True,
    safe=True,
):
    if safe and not _is_markov_matrix(transition_matrix):
        msg = '\'transition_matrix\' should be a square matrix with' \
            'sum of each row equal to one'

        raise ValueError(msg)

    size = len(transition_matrix)
    distribution = np.zeros(size)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= size

    return distribution
