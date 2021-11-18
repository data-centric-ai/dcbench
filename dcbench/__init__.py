"""The dcbench module is a collection for benchmarks that test various apsects
of data preparation and handling in the context of AI workflows."""
# flake8: noqa

from .__main__ import main
from .common import Artifact, Problem, Solution, Table, Task
from .common.artifact import (
    CSVArtifact,
    DataPanelArtifact,
    ModelArtifact,
    VisionDatasetArtifact,
    YAMLArtifact,
)
from .config import config
from .tasks.budgetclean import BudgetcleanProblem
from .tasks.minidata import MiniDataProblem
from .tasks.slice_discovery import SliceDiscoveryProblem
from .version import __version__

__all__ = [
    "Artifact",
    "Problem",
    "Solution",
    "BudgetcleanProblem",
    "MiniDataProblem",
    "SliceDiscoveryProblem",
    "Task",
    "ModelArtifact",
    "YAMLArtifact",
    "DataPanelArtifact",
    "VisionDatasetArtifact",
    "CSVArtifact",
    "config",
]


from .tasks.budgetclean import task as budgetclean_task
from .tasks.minidata import task as minidata_task
from .tasks.slice_discovery import task as slice_discovery_task

tasks = Table(
    [
        minidata_task,
        slice_discovery_task,
        budgetclean_task,
    ]
)
