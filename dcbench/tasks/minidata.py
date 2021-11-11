from typing import Any, Mapping, Sequence

from dcbench.common.artefact import ArtefactSpec, DataPanelArtefact, YAMLArtefact
from dcbench.common.problem import Problem
from dcbench.common.solution import Solution


class MiniDataSolution(Solution):

    # flake8: noqa
    artefact_specs: Mapping[str, ArtefactSpec] = {
        "train_ids": ArtefactSpec(
            artefact_type=YAMLArtefact,
            description=(
                "A list of train example ids from the "
                " ``id`` column of ``train_data``."
            ),
        ),
    }
    task_id: str = "minidata"

    @classmethod
    def from_ids(cls, train_ids: Sequence[str], problem_id: str):
        cls.from_artefacts(
            {"train_ids": train_ids}, attributes={"problem_id": problem_id}
        )


class MiniDataProblem(Problem):
    full_name = "Minimal Data Selection"

    # flake8: noqa
    summary = "Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance."

    artefact_specs: Mapping[str, ArtefactSpec] = {
        "train_data": ArtefactSpec(
            artefact_type=DataPanelArtefact,
            description="A DataPanel of train examples with columns ``id``, "
            "``input``, and ``target``.",
        ),
        "test_data": ArtefactSpec(
            artefact_type=DataPanelArtefact,
            description="A DataPanel of test examples with columns ``id``, "
            "``input``, and ``target``.",
        ),
    }

    solution_class: type = MiniDataSolution

    task_id: str = "minidata"

    def evaluate(self, solution: Solution):

        train_dp = self["train_data"]
        train_ids = solution["train_ids"]

        train_dp = train_dp.lz[train_dp["id"].isin(train_ids)]

        # TODO: Plug unagi in here
        # model = fit(train_dp)
        # score = score(self["test_dp"], model)
        # returnscore
