from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

from dcbench.common.bundle import RelationalBundle

from ..constants import PROBLEMS_DIR
from .artefact import ArtefactContainer

if TYPE_CHECKING:
    from .solution import Solution


class Problem(ArtefactContainer):
    container_id = "problem"
    task_id: str
    result_metrics: Optional[List[str]] = None

    container_dir = PROBLEMS_DIR

    @property
    def solutions(self) -> RelationalBundle[Solution]:
        if self._solutions is None:
            solutions = dict(
                (id, Solution.load(self, id)) for id in Solution.list(self)
            )
            all_result_attributes = set.union(
                *[s.result.keys() for s in solutions.values() if s.result is not None]
            )
            attributes = ["name", "paper", "code"] + [
                "result.%s" % a for a in sorted(all_result_attributes)
            ]
            self._solutions = RelationalBundle(solutions, attributes)
        return self._solutions

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError
