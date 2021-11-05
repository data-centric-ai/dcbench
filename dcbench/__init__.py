"""The dcbench module is a collection for benchmarks that test various apsects
of data preparation and handling in the context of AI workflows."""
# flake8: noqa

from .__main__ import main
from .common import Artefact, Problem, Solution
from .tasks.miniclean import *
from .tasks.slice import SliceDiscoveryProblem
from .version import __version__

problem_classes = Problem.__subclasses__()
