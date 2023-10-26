import random


def _swap(lst, ix_1, ix_2):
    lst[ix_1], lst[ix_2] = lst[ix_2], lst[ix_1]


def _partition(lst, start, end):
    pivot_ix = random.randrange(start, end)
    pivot = lst[pivot_ix]
    _swap(lst, pivot_ix, start)

    bound_ix = start + 1
    swap = False

    for ix in range(start + 1, end):
        if lst[ix] < pivot:
            if swap:
                _swap(lst, bound_ix, ix)

            bound_ix += 1
            continue

        swap = True

    _swap(lst, start, bound_ix - 1)

    return bound_ix - 1


def quicksort(lst, start=None, end=None):
    if start is None:
        start = 0

    if end is None:
        end = len(lst)

    if end - start > 1:
        pivot_ix = _partition(lst, start, end)
        quicksort(lst, start=start, end=pivot_ix)
        quicksort(lst, start=pivot_ix + 1, end=end)


def hello():
    return
