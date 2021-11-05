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
        pass

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError
