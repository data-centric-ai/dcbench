"""Solution to three queriers for general classifier."""
import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd

from .query import Querier


def compute_distances(X_train, X_test):
    dists = np.array(
        [np.sqrt(np.sum((X_train - x_test) ** 2, axis=1)) for x_test in X_test]
    )
    return dists


def majority_vote(A):
    counter = Counter(A)
    major = counter.most_common(1)[0][0]
    return int(major)


class KNN(object):
    """docstring for KNNEvaluator."""

    def __init__(self, n_neighbors=3):
        super(KNN).__init__()
        self.K = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        dists = compute_distances(self.X_train, X_test)
        self.sim = 1 / (1 + dists)
        order = np.argsort(-self.sim, kind="stable", axis=1)
        top_K_idx = order[:, : self.K]
        top_K = self.y_train[top_K_idx]
        pred = np.array([majority_vote(top) for top in top_K])
        return pred

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        acc = np.mean(pred == y_test)
        return acc


class CPClean(object):
    """docstring for CPClean."""

    def __init__(self, K=3, n_jobs=4, random_state=1):
        """Constructor.

        Args:
            K (int): KNN n_neighbors.
            random_state (int): random seed.
            n_jobs (int): number of cpu workers.
        """
        self.K = K
        self.random_state = random_state
        self.n_jobs = n_jobs

    def make_space(self, X_train_repairs, X_val, gt=None):
        sim = np.array(
            [self.compute_similarity(X_train, X_val) for X_train in X_train_repairs]
        )  # shape (#repair, #val, #train)
        space_sim = np.transpose(sim, (2, 0, 1))  # shape (#train, #repair, #val)
        space_X = np.transpose(
            X_train_repairs, (1, 0, 2)
        )  # shape (#row, #repair, #column)
        MM = np.array([sim.min(axis=0), sim.max(axis=0)]).transpose(1, 2, 0)

        space_X = np.around(space_X, decimals=12)
        gt = np.around(gt, decimals=12)

        space = []
        gt_indices = []
        S_val_t = []
        for X, X_gt, S in zip(space_X, gt, space_sim):
            X_unique, X_indices = np.unique(X, axis=0, return_index=True)
            S_unique = S[X_indices]
            gt_id = np.argwhere((X_unique == X_gt).all(axis=1))[0, 0]
            space.append(X_unique)
            S_val_t.append(S_unique)
            gt_indices.append(gt_id)

        S_val = []
        for i in range(len(X_val)):
            s_val = [S[:, i] for S in S_val_t]
            S_val.append(s_val)

        return space, S_val, gt_indices, MM

    def compute_similarity(self, X_train, X_val):
        S_val = np.array(
            [
                1 / (1 + np.sqrt(np.sum((X_train - x_val) ** 2, axis=1)))
                for x_val in X_val
            ]
        )
        return S_val

    def fit(
        self,
        X_train_repairs,
        y_train,
        X_val,
        y_val,
        gt=None,
        X_train_mean=None,
        debugger=None,
        method="cpclean",
        random_state=1,
        sample_size=32,
        restore=False,
    ):
        """Find a world in the space that has the same validation accuracy as
        the ground truth.

        Args:
            X_train_repairs (list): a list of repairs of training set
            y_train (np.array): labels of training set
            gt (np.array): the ground truth index in each row.
        """
        space, S_val, gt_indices, MM = self.make_space(X_train_repairs, X_val, gt=gt)
        if method == "cpclean":
            selection = self.clean(
                S_val, y_train, gt_indices, MM, debugger=debugger, restore=restore
            )
        elif method == "sample_cpclean":
            selection = self.sample_cpclean(
                S_val,
                y_train,
                gt_indices,
                MM,
                debugger=debugger,
                sample_size=sample_size,
            )
        elif method == "sgd_cpclean":
            selection = self.sgd_cpclean(
                S_val,
                y_train,
                gt_indices,
                MM,
                debugger=debugger,
                sample_size=sample_size,
            )
        else:
            selection = self.random_clean(
                S_val, y_train, gt_indices, MM, debugger=debugger
            )

        X_train_clean = deepcopy(X_train_mean)
        for i in selection:
            X_train_clean[i] = gt[i]
        self.classifier = KNN()
        self.classifier.fit(X_train_clean, y_train)

    def score(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)

    def restore_results(self, S_val_pruned, MM, debugger, gt_indices):
        saved_results = pd.read_csv(
            os.path.join(debugger.debug_dir, "details_restore.csv")
        )
        selection = saved_results["selection"].values[1:-2].astype(int).tolist()

        for i in range(len(S_val_pruned)):
            for sel in selection:
                S_val_pruned[i][sel] = [S_val_pruned[i][sel][gt_indices[sel]]]
                MM[i][sel] = [S_val_pruned[i][sel][0], S_val_pruned[i][sel][0]]

        percent_cc = saved_results["percent_cc"].values[0]
        debugger.init_log(percent_cc)
        sel_times = saved_results["time"].values[1:-2].tolist()
        percent_ccs = saved_results["percent_cc"].values[1:-2].tolist()

        n_iter = 1
        for sel, sel_time, percent_cc in zip(selection, sel_times, percent_ccs):
            debugger.log(n_iter, sel, sel_time, percent_cc)
            n_iter += 1
        return selection, n_iter

    def clean(self, S_val, y_train, gt_indices, MM=None, debugger=None, restore=False):
        S_val_pruned = deepcopy(S_val)
        selection = []
        n_iter = 1

        if restore:
            selection, n_iter = self.restore_results(
                S_val_pruned, MM, debugger, gt_indices
            )

        init_querier = Querier(
            self.K,
            S_val_pruned,
            y_train,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        q1_results_pruned, _, before_entropy_pruned = init_querier.run_q1q2(MM=MM)
        MM_pruned = MM

        percent_cc = q1_results_pruned.mean()

        if not restore and debugger is not None:
            debugger.init_log(percent_cc)

        while True:
            tic = time.time()

            # prune
            non_cp_idx = np.argwhere(q1_results_pruned == False).ravel()  # noqa: E712
            S_val_pruned = [S_val_pruned[i] for i in non_cp_idx]
            before_entropy_pruned = before_entropy_pruned[non_cp_idx]
            MM_pruned = [MM_pruned[i] for i in non_cp_idx]

            if len(S_val_pruned) == 0:
                break

            # select
            querier = Querier(
                self.K,
                S_val_pruned,
                y_train,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            sel, after_entropy_pruned = querier.run_q3_select(
                before_entropy_val=before_entropy_pruned, MM=MM_pruned
            )

            if sel is None:
                break

            # update selection
            for i in range(len(S_val_pruned)):
                S_val_pruned[i][sel] = [S_val_pruned[i][sel][gt_indices[sel]]]
                selection.append(sel)

                # update q2 result
                if after_entropy_pruned[i] is not None:
                    before_entropy_pruned[i] = after_entropy_pruned[i][gt_indices[sel]]

                # update MM
                MM_pruned[i][sel] = [S_val_pruned[i][sel][0], S_val_pruned[i][sel][0]]

            # update q1
            q1_results_pruned = querier.run_q1(MM=MM_pruned)

            sel_time = time.time() - tic
            # logging
            percent_cc = (
                len(S_val) - len(S_val_pruned) + sum(q1_results_pruned)
            ) / len(S_val)
            print(
                "Iteration {}, time {}, selection {}, percent_cc {}".format(
                    n_iter, sel_time, sel, percent_cc
                )
            )
            if debugger is not None:
                debugger.log(n_iter, sel, sel_time, percent_cc)

            n_iter += 1

        return selection

    def sample_cpclean(
        self, S_val, y_train, gt_indices, MM=None, debugger=None, sample_size=32
    ):
        S_val = deepcopy(S_val)
        init_querier = Querier(self.K, S_val, y_train, n_jobs=self.n_jobs)
        final_selection = []

        while True:
            q1_results = init_querier.run_q1(MM=MM)
            non_cp_idx = np.argwhere(q1_results == False).ravel()  # noqa: E712

            if len(non_cp_idx) <= sample_size:
                sampled_idx = non_cp_idx
            else:
                np.random.seed(self.random_state)
                sampled_idx = np.random.choice(
                    non_cp_idx, size=sample_size, replace=False
                )

            if len(sampled_idx) == 0:
                break

            S_val_sampled = [S_val[i] for i in sampled_idx]
            MM_sampled = [MM[i] for i in sampled_idx]
            selection = self.clean(
                S_val_sampled, y_train, gt_indices, MM=MM_sampled, debugger=debugger
            )

            final_selection.extend(selection)

            for sel in selection:
                for i in range(len(S_val)):
                    S_val[i][sel] = [S_val[i][sel][gt_indices[sel]]]
                    MM[i][sel] = [S_val[i][sel][0], S_val[i][sel][0]]

    def sgd_cpclean(
        self,
        S_val,
        y_train,
        gt_indices,
        MM=None,
        debugger=None,
        restore=False,
        sample_size=32,
    ):
        S_val_pruned = deepcopy(S_val)
        selection = []
        n_iter = 1

        init_querier = Querier(
            self.K,
            S_val_pruned,
            y_train,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        q1_results_pruned = init_querier.run_q1(MM=MM)
        MM_pruned = MM

        percent_cc = q1_results_pruned.mean()
        debugger.init_log(percent_cc)

        while True:
            tic = time.time()

            # prune
            non_cp_idx = np.argwhere(q1_results_pruned == False).ravel()  # noqa: E712
            S_val_pruned = [S_val_pruned[i] for i in non_cp_idx]
            MM_pruned = [MM_pruned[i] for i in non_cp_idx]
            n_non_cp_val = len(S_val_pruned)

            if n_non_cp_val == 0:
                break

            # sample
            if n_non_cp_val < sample_size:
                sampled_idx = np.arange(n_non_cp_val)
            else:
                np.random.seed(n_iter)
                sampled_idx = np.random.choice(
                    n_non_cp_val, size=sample_size, replace=False
                )

            S_val_sampled = [S_val_pruned[i] for i in sampled_idx]
            MM_sampled = [MM_pruned[i] for i in sampled_idx]

            # select
            querier = Querier(
                self.K,
                S_val_sampled,
                y_train,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            sel, _ = querier.run_q3_select(MM=MM_sampled)

            # update selection
            for i in range(len(S_val_pruned)):
                S_val_pruned[i][sel] = [S_val_pruned[i][sel][gt_indices[sel]]]
                selection.append(sel)
                # update MM
                MM_pruned[i][sel] = [S_val_pruned[i][sel][0], S_val_pruned[i][sel][0]]

            # update q1
            q1_results_pruned = querier.run_q1(MM=MM_pruned)

            sel_time = time.time() - tic

            # logging
            percent_cc = (
                len(S_val) - len(S_val_pruned) + sum(q1_results_pruned)
            ) / len(S_val)
            print(
                "Iteration {}, time {}, selection {}, percent_cc {}".format(
                    n_iter, sel_time, sel, percent_cc
                )
            )
            debugger.log(n_iter, sel, sel_time, percent_cc)

            n_iter += 1

        return selection

    def random_clean(self, S_val, y_train, gt_indices, MM=None, debugger=None):
        S_val_pruned = deepcopy(S_val)

        init_querier = Querier(
            self.K, S_val, y_train, n_jobs=self.n_jobs, random_state=self.random_state
        )
        q1_results_pruned = init_querier.run_q1(MM=MM)
        MM_pruned = MM

        percent_cc = q1_results_pruned.mean()
        debugger.init_log(percent_cc)

        np.random.seed(self.random_state)
        select = [i for i, x in enumerate(S_val[0]) if len(x) > 1]
        np.random.shuffle(select)

        selection = []
        n_iter = 1

        for sel in select:
            tic = time.time()
            # prune
            non_cp_idx = np.argwhere(q1_results_pruned == False).ravel()  # noqa: E712
            S_val_pruned = [S_val_pruned[i] for i in non_cp_idx]
            MM_pruned = [MM_pruned[i] for i in non_cp_idx]

            if len(S_val_pruned) == 0:
                break

            # update selection
            for i in range(len(S_val_pruned)):
                S_val_pruned[i][sel] = [S_val_pruned[i][sel][gt_indices[sel]]]
                selection.append(sel)

                # update MM
                MM_pruned[i][sel] = [S_val_pruned[i][sel][0], S_val_pruned[i][sel][0]]

            # update q1
            q1_results_pruned = init_querier.run_q1(MM=MM_pruned)

            sel_time = time.time() - tic
            # logging
            percent_cc = (
                len(S_val) - len(S_val_pruned) + sum(q1_results_pruned)
            ) / len(S_val)
            print(
                "Iteration {}, time {}, selection {}, percent_cc {}".format(
                    n_iter, sel_time, sel, percent_cc
                )
            )
            debugger.log(n_iter, sel, sel_time, percent_cc)

            n_iter += 1
        return selection
