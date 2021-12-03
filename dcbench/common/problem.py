from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from .artifact import ArtifactContainer
from .result import Result
from .table import Table

if TYPE_CHECKING:
    from .trial import Trial
    from .solution import Solution


class Problem(ArtifactContainer):

    container_type: str = "problem"

    # these class properites must be filled in by problem subclasses
    name: str
    summary: str
    task_id: str
    solution_class: type

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, solution: Solution) -> Result:
        raise NotImplementedError


class ProblemTable(Table):

    def trial(self, solver: Optional[Callable[[Problem], Solution]] = None) -> Trial:
        from .trial import Trial
        return Trial(problems=list(self.values()), solver=solver)
