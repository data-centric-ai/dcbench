from typing import Any
import meerkat as mk

import torch.nn as nn

from dcbench.__main__ import scenarios
from dcbench.common.solution import Solution
from dcbench.common.artefact import (
    DataPanelArtefact,
)
from dcbench.common.problem import Problem


class SliceDiscoveryProblem(Problem):

    # scenario_df = .download()

    artefact_spec = {
        "dataset": DataPanelArtefact,
        "predictions": DataPanelArtefact,
        "activations": DataPanelArtefact
    }

    task_id: str = "slice_discovery"

    # def __init__(self):
    #     self.properties = {
    #         "dataset": "celeba",
    #         "slice_type": "spurious_correlation",
    #         "task": "classification",
    #         "target": "vehicles",
    #     }

    @classmethod
    def list(cls):
        for scenario_id in cls.scenario_df["id"]:
            yield cls.from_id(scenario_id)

    @classmethod
    def from_id(cls, scenario_id: str):
        pass

    def solve(self, **kwargs: Any) -> Solution:
        pass

    def evaluate(self, solution: Solution):
        pass


# Task -> Scenario


# class MiniDatasetProblem(Problem):
#     artefact_spec = {
#         "data_dp": DataPanelArtefact,
#         "model": ModelArtefact
#     }


# scenario = MiniDatasetScenario.from_artefacts(...)
# scenario.solve()
# scenario.evaluate()

# class SliceDiscoveryTask(Task):

# solution.upload()

# SolutionSet.upload()
# ProblemSet should only include one Problem
# ProblemSet(dataset=).download()
# submission: Submission = ProblemSet.solve()
# problem_instance = SliceDiscoveryProblem()

# solution: SliceDiscoverySolution = problem_instance.solve()


# class Submission():

# class Solution(ArtefactContainer):
#     problem_spec: Mapping[str, type]

# class Problem(ArtefactContainer):
#     problem_spec: Mapping[str, type]

#     artefact_spec: Mapping[str, type]


# class SliceDiscoveryProblem(Problem):

#     scenario_df = CsvArtefact(name="slice_discovery_scenarios").download()

#     artefact_spec = {
#         "data_dp": DataPanelArtefact,
#         "model": ModelArtefact
#     }

#     def __init__(self):
#         self.properties = {
#             "dataset": "celeba",
#             "slice_type": "spurious_correlation",
#             "task": "classification",
#             "target": "vehicles"
#         }

#     .groupby()

#     @classmethod
#     def list(cls):
#         for scenario_id in cls.scenario_df["id"]:
#             yield cls.from_id(scenario_id)

#     @classmethod
#     def from_id(cls, scenario_id: str):
#         pass

#     @classmethod
#     def from_artefacts(cls, data_dp: mk.DataPanel, model: nn.Module):
#         pass
