from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ..constants import PROBLEMS_DIR
from .artefact import ArtefactContainer

if TYPE_CHECKING:
    from .solution import Solution


class Problem(ArtefactContainer):
    container_type: str = "problem"
    task_id: str

    container_dir = PROBLEMS_DIR

    solution_class: type

    @abstractmethod
    def evaluate(self, solution: Solution):
        raise NotImplementedError
