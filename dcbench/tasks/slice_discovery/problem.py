from typing import Mapping

import meerkat as mk

from dcbench.common import Problem, Solution
from dcbench.common.artifact import (
    ArtifactSpec,
    DataPanelArtifact,
    ModelArtifact,
    VisionDatasetArtifact,
)

from .metrics import compute_metrics


class SliceDiscoverySolution(Solution):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "pred_slices": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of predicted slice labels with columns `id`"
            " and `pred_slices`.",
        ),
    }

    task_id: str = "slice_discovery"


class SliceDiscoveryProblem(Problem):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "val_predictions": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description=(
                "A DataPanel of the model's predictions with columns `id`,"
                "`target`, and `probs.`"
            ),
        ),
        "test_predictions": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description=(
                "A DataPanel of the model's predictions with columns `id`,"
                "`target`, and `probs.`"
            ),
        ),
        "test_slices": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of the ground truth slice labels with columns "
            " `id`, `slices`.",
        ),
        "activations": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of the model's activations with columns `id`,"
            "`act`",
        ),
        "model": ArtifactSpec(
            artifact_type=ModelArtifact,
            description="A trained PyTorch model to audit.",
        ),
        "base_dataset": ArtifactSpec(
            artifact_type=VisionDatasetArtifact,
            description="A DataPanel representing the base dataset with columns `id` "
            "and `image`.",
        ),
        "clip": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of the image embeddings from OpenAI's CLIP model",
        ),
    }

    task_id: str = "slice_discovery"

    def solve(self, pred_slices_dp: mk.DataPanel) -> SliceDiscoverySolution:
        if ("id" not in pred_slices_dp) or ("pred_slices" not in pred_slices_dp):
            raise ValueError(
                f"DataPanel passed to {self.__class__.__name__} must include columns "
                "`id` and `pred_slices`"
            )

        return SliceDiscoverySolution.from_artifacts(
            artifacts={"pred_slices": pred_slices_dp},
            attributes={"problem_id": self.id},
        )

    def evaluate(self, solution: SliceDiscoverySolution) -> dict:
        dp = mk.merge(self["test_slices"], solution["pred_slices"], on="id")
        result = compute_metrics(dp["slices"], dp["pred_slices"])
        return result[["precision_at_10", "precision_at_25", "auroc"]].mean().to_dict()
