import numpy as np


def random_select(dirty_rows):
    """Random select one from dirty rows."""
    sel = np.random.choice(dirty_rows)
    return sel


def compute_avg_dirty_entropies(after_entropies, dirty_rows):
    avg_entropies = []
    for i in dirty_rows:
        if after_entropies[i] is None:
            avg_entropies.append(np.nan)
        else:
            avg_entropies.append(sum(after_entropies[i]) / len(after_entropies[i]))
    return np.array(avg_entropies)


def min_entropy_expected(
    after_entropy_val, dirty_rows, before_entropies_val, n_jobs=4
):  # already checked
    """
    Args:
        ac_counters_val (list): Counts after clean for each cell for each test example
            (only for dirty rows)
        dirty rows (list): indices of dirty rows
    """
    avg_entropies_val = [
        compute_avg_dirty_entropies(ae, dirty_rows) for ae in after_entropy_val
    ]

    for i in range(len(avg_entropies_val)):
        mask = np.isnan(avg_entropies_val[i])
        avg_entropies_val[i][mask] = before_entropies_val[i]

    avg_entropies_val = np.array(avg_entropies_val)
    info_gain = (before_entropies_val.reshape(-1, 1) - avg_entropies_val).mean(axis=0)
    info_gain[info_gain == 0] = float("-inf")
    max_idx = np.argmax(info_gain)
    sel = dirty_rows[max_idx]
    return sel


def entropy_expected(
    after_entropy_val, dirty_rows, before_entropies_val, n_jobs=4
):  # already checked
    """
    Args:
        ac_counters_val (list): Counts after clean for each cell for each test example
            (only for dirty rows)
        dirty rows (list): indices of dirty rows
    """
    avg_entropies_val = [
        compute_avg_dirty_entropies(ae, dirty_rows) for ae in after_entropy_val
    ]

    for i in range(len(avg_entropies_val)):
        mask = np.isnan(avg_entropies_val[i])
        avg_entropies_val[i][mask] = before_entropies_val[i]

    avg_entropies_val = np.array(avg_entropies_val)
    info_gain = (before_entropies_val.reshape(-1, 1) - avg_entropies_val).mean(axis=0)
    info_gain[info_gain == 0] = float("-inf")
    return info_gain
