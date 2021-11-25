import meerkat as mk
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn

import dcbench
from dcbench.common.artifact import DataPanelArtifact
from dcbench.common.table import Table
from dcbench.common.task import Task
from dcbench.tasks.slice_discovery.metrics import compute_metrics


def test_problems():
    slice_discovery = dcbench.tasks["slice_discovery"]
    assert isinstance(slice_discovery, Task)
    assert isinstance(slice_discovery.problems, Table)


def test_problem():
    slice_discovery = dcbench.tasks["slice_discovery"]
    problem = slice_discovery.problems["p_117634"]

    for name in ["test_predictions", "val_predictions", "test_slices", "activations"]:
        out = problem[name]
        assert isinstance(out, mk.DataPanel)

    out = problem["model"]
    assert isinstance(out, nn.Module)

    problem.slice_category


def test_artifacts():
    slice_discovery = dcbench.tasks["slice_discovery"]
    problem = slice_discovery.problems["p_117634"]
    artifacts = problem.artifacts
    assert isinstance(artifacts, dict)
    for name in ["test_predictions", "val_predictions", "test_slices", "activations"]:
        artifact = artifacts[name]
        assert isinstance(artifact, DataPanelArtifact)
        assert not artifact.is_downloaded

        artifact.download()

        assert artifact.is_downloaded


def test_metrics():
    slices = np.array([[0, 1], [0, 0], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]])

    pred_slices = np.array(
        [[0, 10, 3], [0, 0, 4], [1, 0, 0], [0, 10, 0], [0, 0, 0], [1, 0, 0], [0, 10, 0]]
    )

    metrics = compute_metrics(pred_slices=pred_slices, slices=slices)

    assert metrics["auroc"][0] == roc_auc_score(slices[:, 0], pred_slices[:, 0])
    assert metrics["auroc"][1] == roc_auc_score(slices[:, 1], pred_slices[:, 1])
