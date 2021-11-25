"""Solution to three queriers for KNN classifier."""
import numpy as np

from .algorithm.min_max import min_max_val
from .algorithm.select import min_entropy_expected, random_select
from .algorithm.sort_count import sort_count_after_clean_multi, sort_count_dp_multi
from .algorithm.utils import compute_entropy_by_counts

# from .algorithm.sort_count import


class Querier(object):
    """docstring for Querier."""

    def __init__(self, K, S_val, y_train, n_jobs=4, random_state=1):
        """Constructor.

        Args:
            K (int): KNN hyper-parameter
            space (list of list of np.array): each row contains a list of candidates
            (repairs) for one example
            y_train (np.array): labels of training set
            X_val (np.array): features of test set
            y_val (np.array): list of test set
            gt_indices (np.array): the ground truth index in each row
        """
        self.K = K
        self.S_val = S_val
        self.y_train = y_train
        self.classes = list(set(y_train))
        self.n_jobs = n_jobs
        self.random_state = random_state

    def run_q1(self, return_preds=False, MM=None):
        """Solution for q1.

        Return:
            q1_results (list of boolean): for each example in test set, whether it can
                be CP'ed.
        """
        if MM is None:
            MM = []
            for S in self.S_val:
                mm = np.array([[min(s), max(s)] for s in S])
                MM.append(mm)

        q1_results, _, pred_sets = min_max_val(MM, self.y_train, self.K)

        if return_preds:
            return q1_results, pred_sets
        else:
            return q1_results

    def run_q2(self, return_entropy=False, MM=None):
        """Solution for q2.

        Return:
            results (list of dict): the number of worlds supporting each label for each
                example in test set.
        """
        q2_results = sort_count_dp_multi(
            self.S_val, self.y_train, self.K, n_jobs=self.n_jobs, MM=MM
        )
        if return_entropy:
            entropies_val = np.array(
                [compute_entropy_by_counts(counts) for counts in q2_results]
            )
            return q2_results, entropies_val
        else:
            return q2_results

    def run_q1q2(self, MM=None, return_entropy=True):
        q1_results, cp_preds = self.run_q1(return_preds=True, MM=MM)

        cp_idx = [i for i, cp in enumerate(q1_results) if cp]
        not_cp_idx = [i for i, cp in enumerate(q1_results) if not cp]

        q2_cp = []
        for i in cp_idx:
            assert len(cp_preds[i]) == 1
            pred = cp_preds[i][0]
            res = {c: 0 for c in self.classes}
            res[pred] = 1
            q2_cp.append(res)

        S_val_no_cp = [self.S_val[i] for i in not_cp_idx]
        q2_no_cp = sort_count_dp_multi(
            S_val_no_cp, self.y_train, self.K, n_jobs=self.n_jobs
        )
        q2_results = self.merge_result([q2_cp, q2_no_cp], [cp_idx, not_cp_idx])

        if return_entropy:
            entropies_val = np.array(
                [compute_entropy_by_counts(counts) for counts in q2_results]
            )
            return q1_results, q2_results, entropies_val

        return q1_results, q2_results

    def run_q3_select(self, method="cpclean", before_entropy_val=None, MM=None):
        dirty_rows = [i for i, x in enumerate(self.S_val[0]) if len(x) > 1]

        if method == "cpclean":
            assert len(self.S_val) == len(MM)
            after_entropy_val = sort_count_after_clean_multi(
                self.S_val, self.y_train, self.K, self.n_jobs, MM=MM
            )

            # extract counters for dirty rows
            if before_entropy_val is None:
                _, before_entropy_val = self.run_q2(MM=MM, return_entropy=True)

            sel = min_entropy_expected(
                after_entropy_val, dirty_rows, before_entropy_val, n_jobs=self.n_jobs
            )

            after_entropy_val_sel = [ae[sel] for ae in after_entropy_val]

            return sel, after_entropy_val_sel
        elif method == "random":
            sel = random_select(dirty_rows)
            return sel, None
        else:
            raise Exception("Wrong method")

    def merge_result(self, results, indices):
        res_a, res_b = results
        idx_a, idx_b = indices

        l = len(idx_a) + len(idx_b)
        merge = [None for _ in range(l)]
        for res, i in zip(res_a, idx_a):
            merge[i] = res
        for res, i in zip(res_b, idx_b):
            merge[i] = res

        for res in merge:
            assert res is not None
        return merge
