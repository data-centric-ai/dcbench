from typing import Any, Mapping

from dcbench.common.artefact import ArtefactSpec, DataPanelArtefact
from dcbench.common.problem import Problem
from dcbench.common.solution import Solution


class MiniDataProblem(Problem):

    # flake8: noqa
    summary = "Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance."

    artefact_specs: Mapping[str, ArtefactSpec] = {
        "train_data": ArtefactSpec(
            artefact_type=DataPanelArtefact,
            description="""Training data.""",
        ),
        "test_data": ArtefactSpec(
            artefact_type=DataPanelArtefact, description="""Testing data."""
        ),
    }

    task_id: str = "minidata"

    def solve(self, **kwargs: Any) -> Solution:
        pass

    def evaluate(self, solution: Solution):
        pass
