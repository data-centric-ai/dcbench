from typing import Mapping

import meerkat as mk

from dcbench.common import Problem, Solution, Task
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
        "predictions": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description=(
                "A DataPanel of the model's predictions with columns `id`,"
                "`target`, and `probs.`"
            ),
        ),
        "slices": ArtifactSpec(
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
    }

    task_id: str = "slice_discovery"

    def solve(self, pred_slices_dp: mk.DataPanel) -> SliceDiscoverySolution:
        if ("id" not in pred_slices_dp) or ("pred_slices_dp" not in pred_slices_dp):
            raise ValueError(
                f"DataPanel passed to {self.__class__.__name__} must include columns "
                ""
            )
        return SliceDiscoverySolution.from_artifacts(
            artifacts={"pred_slices_dp": pred_slices_dp}
        )

    def evaluate(self, solution: SliceDiscoverySolution) -> dict:
        dp = mk.merge(self["slices"], solution["pred_slices"], on="id")
        result = compute_metrics(dp=dp)
        return result[["precision_at_10", "precision_at_25", "auroc"]].mean().to_dict()


task = Task(
    task_id="slice_discovery",
    name="Slice Discovery",
    summary=(
        "Machine learnings models that achieve high overall accuracy often make "
        " systematic erors on important subgroups (or *slices*) of data. When working  "
        " with high-dimensional inputs (*e.g.* images, audio) where data slices are  "
        " often unlabeled, identifying underperforming slices is challenging. In "
        " this task, we'll develop automated slice discovery methods that mine "
        " unstructured data for underperforming slices."
    ),
    problem_class=SliceDiscoveryProblem,
    solution_class=SliceDiscoverySolution,
    baselines=None,
)
