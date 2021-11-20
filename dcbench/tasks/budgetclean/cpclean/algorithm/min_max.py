import numpy as np

from .utils import majority_vote


def min_max(mm, y, K):
    """MinMax algorithm.

    Given a similarity matrix, return whether it is CP or not and the
    best scenario for each label.  mm (np.array): shape Nx2. the min and
    max similarity of each row.  y (list): labels  K (int): KNN
    hyperparameter
    """
    assert len(set(y)) == 2
    pred_set = set()
    best_scenarios = {}

    for c in [0, 1]:
        best_scenario = np.zeros(len(y))
        mask = y == c

        # set min max
        best_scenario[mask] = mm[:, 1][mask]
        best_scenario[(mask == False)] = mm[:, 0][(mask == False)]  # noqa: E712

        # run KNN
        order = np.argsort(-best_scenario, kind="stable")
        top_K = y[order][:K]

        pred = majority_vote(top_K)

        if pred == c:
            pred_set.add(c)

        best_scenarios[c] = (best_scenario, pred)

    is_cc = len(pred_set) == 1

    return is_cc, best_scenarios, list(pred_set)


def min_max_val(MM, y, K):
    q1_results = []
    scenarios = []
    cc_preds = []
    for mm in MM:
        cc, sc, pred = min_max(mm, y, K)
        q1_results.append(cc)
        scenarios.append(sc)
        cc_preds.append(pred)

    q1_results = np.array(q1_results)
    return q1_results, scenarios, cc_preds
