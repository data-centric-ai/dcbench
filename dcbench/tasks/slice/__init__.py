from typing import Any, Mapping

from dcbench.common.artefact import (
    ArtefactSpec,
    DataPanelArtefact,
    VisionDatasetArtefact,
)
from dcbench.common.problem import Problem
from dcbench.common.solution import Solution


class SliceDiscoveryProblem(Problem):

    summary = (
        "Machine learnings models that achieve high overall accuracy often make "
        " systematic erors on important subgroups (or *slices*) of data. When working  "
        " with high-dimensional inputs (*e.g.* images, audio) where data slices are  "
        " often unlabeled, identifying underperforming slices is a challenging. In "
        " this task, we'll develop automated slice discovery methods that mine "
        " unstructured data for underperforming slices."
    )

    artefact_specs: Mapping[str, ArtefactSpec] = {
        "predictions": ArtefactSpec(
            artefact_type=DataPanelArtefact,
            description=(
                "A Datapanel of the model's predictions with columns `id`,"
                "`target`, and `probs.`"
            ),
        ),
        "slices": ArtefactSpec(
            artefact_type=DataPanelArtefact,
            description="""
                A datapanel containing ground truth slice labels """,
        ),
        "activations": ArtefactSpec(artefact_type=DataPanelArtefact, description=""),
        "base_dataset": ArtefactSpec(
            artefact_type=VisionDatasetArtefact, description="A base dataset"
        ),
    }

    task_id: str = "slice_discovery"

    @classmethod
    def from_id(cls, scenario_id: str):
        pass

    def solve(self, **kwargs: Any) -> Solution:
        pass

    def evaluate(self, solution: Solution):
        pass
