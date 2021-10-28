from __future__ import annotations
import os
import urllib.parse

from abc import abstractmethod
from typing import Mapping, Optional, List, Any, Type, TYPE_CHECKING


from dcbench.common.bundle import RelationalBundle

from .artefact import Artefact, ArtefactContainer, ArtefactInstance, ArtefactBundle
from ..constants import (
    LOCAL_DIR,
    SCENARIOS_DIR,
    PUBLIC_REMOTE_URL,

    HIDDEN_ARTEFACTS_URL,
)
from .download_utils import download_and_extract_archive
if TYPE_CHECKING:
    from .solution import Solution


class Problem(ArtefactContainer):

    id: Optional[str] = None
    result_metrics: Optional[List[str]] = None

    def __init__(self, artefacts: Mapping[str, Artefact], working_dir: str = None):
        super().__init__(artefacts=artefacts)
        self.working_dir: str = working_dir or LOCAL_DIR

    @property
    def location(self) -> str:
        return os.path.join(self.working_dir, SCENARIOS_DIR, self.id)
 
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
        return cls.scenarios.keys()

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        pass

    @abstractmethod
    def evaluate(self, solution: Solution) -> Result:
        pass

