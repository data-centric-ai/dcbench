import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def precision_at_k(slice: np.ndarray, pred_slice: np.ndarray, k: int = 25):
    return precision_score(
        slice, rankdata(-pred_slice, method="ordinal") <= k, zero_division=0
    )


def recall_at_k(slice: np.ndarray, pred_slice: np.ndarray, k: int = 25):
    return recall_score(slice, rankdata(-pred_slice) <= k, zero_division=0)


PRECISION_K = [10, 25, 100]
RECALL_K = [50, 100, 200]


def compute_metrics(slices: np.ndarray, pred_slices: np.ndarray) -> dict:
    """[summary]

    Args:
        slices (np.ndarray): [description]
        pred_slices (np.ndarray): [description]

    Returns:
        dict: [description]
    """

    pred_slice = pred_slices.argmax(axis=-1)
    no_nan_preds = not np.isnan(pred_slices).any()

    rows = []
    for slice_idx in range(slices.shape[1]):

        df = pd.DataFrame(
            [
                {
                    "pred_slice_idx": pred_slice_idx,
                    "slice_idx": slice_idx,
                    "auroc": roc_auc_score(
                        slices[:, slice_idx], pred_slices[:, pred_slice_idx]
                    )
                    if len(np.unique(slices[:, slice_idx])) > 1 and no_nan_preds
                    else np.nan,
                    **{
                        f"precision_at_{k}": precision_at_k(
                            slices[:, slice_idx],
                            pred_slices[:, pred_slice_idx],
                            k=k,
                        )
                        if len(np.unique(slices[:, slice_idx])) > 1 and no_nan_preds
                        else np.nan
                        for k in PRECISION_K
                    },
                    **{
                        f"recall_at_{k}": recall_at_k(
                            slices[:, slice_idx],
                            pred_slices[:, pred_slice_idx],
                            k=k,
                        )
                        if len(np.unique(slices[:, slice_idx])) > 1 and no_nan_preds
                        else np.nan
                        for k in RECALL_K
                    },
                    "recall": recall_score(
                        slices[:, slice_idx],
                        (pred_slice == pred_slice_idx).astype(int),
                    )
                    if no_nan_preds
                    else np.nan,
                    "precision": precision_score(
                        slices[:, slice_idx],
                        (pred_slice == pred_slice_idx).astype(int),
                        zero_division=0,
                    )
                    if no_nan_preds
                    else np.nan,
                }
                for pred_slice_idx in range(pred_slices.shape[1])
            ]
        )
        # take the predicted slice idx with the maximum auroc
        rows.append(df.loc[df["auroc"].idxmax()].to_dict())

    return pd.DataFrame(rows)
