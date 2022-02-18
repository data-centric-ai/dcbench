from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from .artifact_container import ArtifactContainer
from .result import Result
from .table import Table

if TYPE_CHECKING:
    from .solution import Solution
    from .trial import Trial


class Problem(ArtifactContainer):
    """A logical collection of :class:`Artifact`s and "attributes" that correspond to a
    specific problem to be solved.

    See the walkthrough section on :ref:`problem-intro` for more information.
    """

    container_type: str = "problem"

    # these class properties must be filled in by problem subclasses
    name: str
    summary: str
    task_id: str
    solution_class: type

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, solution: Solution) -> Result:
        raise NotImplementedError()


class ProblemTable(Table):
    def trial(self, solver: Optional[Callable[[Problem], Solution]] = None) -> Trial:
        from .trial import Trial

        return Trial(problems=list(self.values()), solver=solver)
