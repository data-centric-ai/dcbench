import meerkat as mk
import numpy as np

from ...common.baseline import baseline
from .problem import SliceDiscoveryProblem, SliceDiscoverySolution


@baseline(
    id="confusion_sdm",
    summary=(
        "A simple slice discovery method that returns a slice corresponding to "
        "each cell of the confusion matrix."
    ),
)
def confusion_sdm(problem: SliceDiscoveryProblem) -> SliceDiscoverySolution:
    """A simple slice discovery method that returns a slice corresponding to each cell
     of the confusion matrix. For example, for a binary prediction task, this sdm will
     return 4 slices corresponding to true positives, false positives, true negatives
     and false negatives.

    Args:
        problem (SliceDiscoveryProblem): The slice discovery problem.

    Returns:
        SliceDiscoverySolution: The predicted slices.
    """

    # the budget of predicted slices allowed by the problem
    n_pred_slices: int = problem.n_pred_slices

    # the only aritfact used by this simple baseline is the model predictions
    predictions_dp = problem["predictions"]

    pred_slices = np.stack(
        [
            (predictions_dp["target"] == target_idx)
            * (predictions_dp["probs"][:, pred_idx]).numpy()
            for target_idx in range(predictions_dp["probs"].shape[1])
            for pred_idx in range(predictions_dp["probs"].shape[1])
        ],
        axis=-1,
    )
    if pred_slices.shape[1] > n_pred_slices:
        raise ValueError(
            "ConfusionSDM is not configured to return enough slices to "
            "capture the full confusion matrix."
        )

    if pred_slices.shape[1] < n_pred_slices:
        # fill in the other predicted slices with zeros
        pred_slices = np.concatenate(
            [
                pred_slices,
                np.zeros((pred_slices.shape[0], n_pred_slices - pred_slices.shape[1])),
            ],
            axis=1,
        )

    return problem.solve(
        pred_slices_dp=mk.DataPanel(
            {"id": predictions_dp["id"], "pred_slices": pred_slices}
        )
    )
