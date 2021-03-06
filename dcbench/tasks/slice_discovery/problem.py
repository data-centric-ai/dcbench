from typing import Mapping

import meerkat as mk

from dcbench.common import Problem, Solution
from dcbench.common.artifact import (
    DataPanelArtifact,
    ModelArtifact,
    VisionDatasetArtifact,
)
from dcbench.common.artifact_container import ArtifactSpec
from dcbench.common.table import AttributeSpec



class SliceDiscoverySolution(Solution):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "pred_slices": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of predicted slice labels with columns `id`"
            " and `pred_slices`.",
        ),
    }

    attribute_specs = {
        "problem_id": AttributeSpec(
            description="A unique identifier for this problem.",
            attribute_type=str,
        ),
        "slicer_class": AttributeSpec(
            description="The ",
            attribute_type=type,
        ),
        "slicer_config": AttributeSpec(
            description="The configuration for the slicer.",
            attribute_type=dict,
        ),
        "embedding_column": AttributeSpec(
            description="The column name of the embedding.",
            attribute_type=str,
        ),
    }

    task_id: str = "slice_discovery"

    @property
    def problem(self):
        from dcbench import tasks
        return tasks["slice_discovery"].problems[self.problem_id]

    def merge(self) -> mk.DataPanel:
        return self["pred_slices"].merge(
            self.problem.merge(split="test", slices=True), on="id", how="left"
        )

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

    attribute_specs = {
        "n_pred_slices": AttributeSpec(
            description="The number of slice predictions that each slice discovery "
            "method can return.",
            attribute_type=int,
        ),
        "slice_category": AttributeSpec(
            description="The type of slice .", attribute_type=str
        ),
        "target_name": AttributeSpec(
            description="The name of the target column in the dataset.",
            attribute_type=str,
        ),
        "dataset": AttributeSpec(
            description="The name of the dataset being audited.",
            attribute_type=str,
        ),
        "alpha": AttributeSpec(
            description="The alpha parameter for the AUC metric.",
            attribute_type=float,
        ),
        "slice_names": AttributeSpec(
            description="The names of the slices in the dataset.",
            attribute_type=list,
        ),
    }

    task_id: str = "slice_discovery"

    def merge(self, split="val", slices: bool = False):
        base_dataset = self["base_dataset"]
        base_dataset = base_dataset[[c for c in base_dataset.columns if c != "split"]]
        dp = self[f"{split}_predictions"].merge(
            base_dataset, on="id", how="left"
        )
        if slices:
            dp = dp.merge(self[f"{split}_slices"], on="id", how="left")
        return dp

    def solve(self, pred_slices_dp: mk.DataPanel) -> SliceDiscoverySolution:
        if ("id" not in pred_slices_dp) or ("pred_slices" not in pred_slices_dp):
            raise ValueError(
                f"DataPanel passed to {self.__class__.__name__} must include columns "
                "`id` and `pred_slices`"
            )

        return SliceDiscoverySolution(
            artifacts={"pred_slices": pred_slices_dp},
            attributes={"problem_id": self.id},
        )


    def evaluate(self):
        pass
