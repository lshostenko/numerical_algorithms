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


def _quicksort(lst, start, end):
    if end - start > 1:
        pivot_ix = _partition(lst, start, end)
        _quicksort(lst, start, pivot_ix)
        _quicksort(lst, pivot_ix + 1, end)


def quicksort(lst):
    _quicksort(lst, 0, len(lst))
