from dcbench.common import Task

from .baselines import confusion_sdm, domino_sdm
from .problem import SliceDiscoveryProblem, SliceDiscoverySolution

__all__ = [
    "confusion_sdm",
    "SliceDiscoveryProblem",
    "SliceDiscoverySolution",
]

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
