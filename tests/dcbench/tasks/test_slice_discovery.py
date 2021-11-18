import numpy as np
from sklearn.metrics import roc_auc_score

from dcbench.tasks.slice_discovery import compute_metrics


def test_metrics():
    slices = np.array([[0, 1], [0, 0], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]])

    pred_slices = np.array(
        [[0, 10, 3], [0, 0, 4], [1, 0, 0], [0, 10, 0], [0, 0, 0], [1, 0, 0], [0, 10, 0]]
    )

    metrics = compute_metrics(pred_slices=pred_slices, slices=slices)

    assert metrics["auroc"][0] == roc_auc_score(slices[:, 0], pred_slices[:, 0])
    assert metrics["auroc"][1] == roc_auc_score(slices[:, 1], pred_slices[:, 1])


def test_():
    pass
