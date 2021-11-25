import random
import time

import numpy as np

from ...common.baseline import baseline
from .common import Preprocessor
from .cpclean.algorithm.select import entropy_expected
from .cpclean.algorithm.sort_count import sort_count_after_clean_multi
from .cpclean.clean import CPClean, Querier

# avoid circular dependency
from .problem import BudgetcleanProblem, BudgetcleanSolution


@baseline(
    id="random_clean", summary="Always selects a random subset of the data to clean."
)
def random_clean(problem: BudgetcleanProblem, seed: int = 1337) -> BudgetcleanSolution:
    size = len(problem["X_train_dirty"])
    budget = int(problem.attributes["budget"] * size)
    random.seed(seed)
    selection = random.sample(range(size), budget)
    idx_selected = [idx in selection for idx in range(size)]
    return problem.solve(idx_selected=idx_selected)


@baseline(
    id="cp_clean",
    summary=(
        "Perform the selection using the CPClean algorithm which aims to "
        " maximize expected information gain."
    ),
)
def cp_clean(
    problem: BudgetcleanProblem, seed: int = 1337, n_jobs=8, kparam=3
) -> BudgetcleanSolution:

    size = len(problem["X_train_dirty"])
    budget = int(problem.attributes["budget"] * size)

    X_train_dirty = problem["X_train_dirty"]
    X_train_clean = problem["X_train_clean"]
    y_train = problem["y_train"]
    X_val = problem["X_val"]

    # Compute number of repairs.
    def length(x):
        if isinstance(x, list):
            return len(x)
        return 0

    num_repairs = X_train_dirty.applymap(length).max().max()

    # Reconstruct separate repair data frames.
    X_train_repairs = {}
    for i in range(num_repairs):

        def getitem(x):
            if isinstance(x, list):
                return x[i] if len(x) > i else None
            return x

        X_train_repairs["repair%02d" % i] = X_train_dirty.applymap(getitem)

    # Replace lists with None values.
    def clearlists(x):
        if isinstance(x, list):
            return None
        return x

    X_train_dirty = X_train_dirty.applymap(clearlists)

    # Preprocess data.
    preprocessor = Preprocessor()
    preprocessor.fit(X_train_dirty, y_train)
    X_train_clean, y_train = preprocessor.transform(X_train_clean, y_train)
    X_val = preprocessor.transform(X_val)
    for name, X in X_train_repairs.items():
        X_train_repairs[name] = preprocessor.transform(X=X)

    d_train_repairs = []
    repair_methods = sorted(X_train_repairs.keys())
    X_train_repairs_sorted = [X_train_repairs[m] for m in repair_methods]
    for X in X_train_repairs_sorted:
        d = np.sum((X - X_train_clean) ** 2, axis=1)
        d_train_repairs.append(d)
    d_train_repairs = np.array(d_train_repairs).T
    gt_indices = np.argmin(d_train_repairs, axis=1)
    X_train_gt = []

    for i, gt_i in enumerate(gt_indices):
        X_train_gt.append(X_train_repairs_sorted[gt_i][i])
    X_train_gt = np.array(X_train_gt)

    # Perform cleaning using CPClean.
    cleaner = CPClean(K=kparam, n_jobs=n_jobs, random_state=seed)
    X_train_repairs = np.array([X_train_repairs[k] for k in X_train_repairs])
    space, S_val, gt_indices, MM = cleaner.make_space(
        X_train_repairs, X_val, gt=X_train_gt
    )
    init_querier = Querier(kparam, S_val, y_train, n_jobs=n_jobs, random_state=seed)

    start = time.time()
    after_entropy_val = sort_count_after_clean_multi(
        S_val, y_train, kparam, n_jobs, MM=MM
    )
    end = time.time()
    print("sort_count_after_clean_multi", end - start)

    start = time.time()
    _, before_entropy_val = init_querier.run_q2(MM=MM, return_entropy=True)
    end = time.time()
    print("run_q2", end - start)

    dirty_rows = [i for i, x in enumerate(S_val[0]) if len(x) > 1]
    info_gain = entropy_expected(
        after_entropy_val, dirty_rows, before_entropy_val, n_jobs=n_jobs
    )
    selection = np.argpartition(info_gain, -budget)[-budget:]

    # Produce solution.
    idx_selected = [idx in selection for idx in range(len(X_train_dirty))]
    return problem.solve(idx_selected=idx_selected)
