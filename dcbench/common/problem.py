from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .artifact import ArtifactContainer

if TYPE_CHECKING:
    from .solution import Solution


class Problem(ArtifactContainer):

    container_type: str = "problem"

    # these class properites must be filled in by problem subclasses
    name: str
    summary: str
    task_id: str
    solution_class: type

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError
