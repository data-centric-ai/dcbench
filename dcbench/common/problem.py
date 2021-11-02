from __future__ import annotations
import os
import urllib.parse

from abc import abstractmethod
from typing import Mapping, Optional, List, Any, Type, TYPE_CHECKING


from dcbench.common.bundle import RelationalBundle

from .artefact import Artefact, ArtefactContainer
from ..constants import (
    LOCAL_DIR,
    PROBLEMS_DIR,
    PUBLIC_REMOTE_URL,
    HIDDEN_ARTEFACTS_URL,
)
from .download_utils import download_and_extract_archive

if TYPE_CHECKING:
    from .solution import Solution


class Problem(ArtefactContainer):

    id: Optional[str] = None
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

    @classmethod
    def list(cls) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, solution: Solution) -> Result:
        raise NotImplementedError
