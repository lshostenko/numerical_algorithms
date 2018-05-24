def _merge(list_1, list_2):
    merged = []
    ix_1, ix_2 = 0, 0

    while all([list_1[ix_1:], list_2[ix_2:]]):
        val_1, val_2 = list_1[ix_1], list_2[ix_2]

        if val_2 < val_1:
            merged.append(val_2)
            ix_2 += 1
            continue

        merged.append(val_1)
        ix_1 += 1

    merged.extend(list_1[ix_1:])
    merged.extend(list_2[ix_2:])

    return merged


def mergesort(values):
    if len(values) < 2:
        return values

    mid_ix = len(values) // 2
    list_1 = mergesort(values[:mid_ix])
    list_2 = mergesort(values[mid_ix:])

    return _merge(list_1, list_2)
