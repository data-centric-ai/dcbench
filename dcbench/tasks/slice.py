from typing import Mapping

from dcbench.common.artifact import (
    ArtifactSpec,
    DataPanelArtifact,
    ModelArtifact,
    VisionDatasetArtifact,
)
from dcbench.common.problem import Problem
from dcbench.common.solution import Solution


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
    full_name = "Slice Discovery"
    summary = (
        "Machine learnings models that achieve high overall accuracy often make "
        " systematic erors on important subgroups (or *slices*) of data. When working  "
        " with high-dimensional inputs (*e.g.* images, audio) where data slices are  "
        " often unlabeled, identifying underperforming slices is a challenging. In "
        " this task, we'll develop automated slice discovery methods that mine "
        " unstructured data for underperforming slices."
    )

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
            " `id`, `target`, and `probs.`",
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

    solution_class = SliceDiscoverySolution

    @classmethod
    def from_id(cls, scenario_id: str):
        pass

    def evaluate(self, solution: Solution):
        pass
