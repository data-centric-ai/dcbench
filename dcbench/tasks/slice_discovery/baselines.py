import meerkat as mk
import numpy as np
from sklearn.decomposition import PCA

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
    """A simple slice discovery method that returns a slice corresponding to
    each cell of the confusion matrix. For example, for a binary prediction
    task, this sdm will return 4 slices corresponding to true positives, false
    positives, true negatives and false negatives.

    Args:
        problem (SliceDiscoveryProblem): The slice discovery problem.

    Returns:
        SliceDiscoverySolution: The predicted slices.
    """

    # the budget of predicted slices allowed by the problem
    n_pred_slices: int = problem.n_pred_slices

    # the only aritfact used by this simple baseline is the model predictions
    predictions_dp = problem["test_predictions"]

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


@baseline(id="domino_sdm", summary=("An error aware mixture model."))
def domino_sdm(problem: SliceDiscoveryProblem) -> SliceDiscoverySolution:
    from .domino import DominoMixture

    # the budget of predicted slices allowed by the problem
    n_pred_slices: int = problem.n_pred_slices

    mm = DominoMixture(
        n_components=25,
        weight_y_log_likelihood=10,
        init_params="error",
        covariance_type="diag",
    )

    dp = mk.merge(problem["val_predictions"], problem["clip"], on="id")
    emb = dp["emb"]

    pca = PCA(n_components=128)
    pca.fit(X=emb)
    pca.fit(X=emb)
    emb = pca.transform(X=emb)

    mm.fit(X=emb, y=dp["target"], y_hat=dp["probs"])

    slice_cluster_indices = (
        -np.abs((mm.y_probs[:, 1] - mm.y_hat_probs[:, 1]))
    ).argsort()[:n_pred_slices]

    dp = mk.merge(problem["test_predictions"], problem["clip"], on="id")
    emb = dp["emb"]
    clusters = mm.predict_proba(
        X=pca.transform(dp["emb"]), y=dp["target"], y_hat=dp["probs"]
    )

    pred_slices = clusters[:, slice_cluster_indices]

    return problem.solve(
        pred_slices_dp=mk.DataPanel({"id": dp["id"], "pred_slices": pred_slices})
    )
