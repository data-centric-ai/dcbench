"""The dcbench module is a collection for benchmarks that test various apsects
of data preparation and handling in the context of AI workflows."""
# flake8: noqa

from .__main__ import main
from .common import Artifact, Problem, Solution
from .common.artifact import (
    CSVArtifact,
    DataPanelArtifact,
    VisionDatasetArtifact,
    YAMLArtifact,
)
from .config import config
from .tasks.miniclean.problem import MinicleanProblem, MiniCleanSolution
from .tasks.minidata import MiniDataProblem, MiniDataSolution
from .tasks.slice import SliceDiscoveryProblem, SliceDiscoverySolution
from .version import __version__

__all__ = [
    "Problem",
    "SliceDiscoveryProblem",
    "SliceDiscoverySolution",
    "MiniDataProblem",
    "MiniDataSolution",
    "MinicleanProblem",
    "MiniCleanSolution",
    "Artifact",
    "YAMLArtifact",
    "DataPanelArtifact",
    "VisionDatasetArtifact",
    "CSVArtifact",
    "config",
]

tasks = [MiniDataProblem, SliceDiscoveryProblem, MinicleanProblem]
