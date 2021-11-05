import json
import os
import random
import string
from typing import Any, Iterator, List, Mapping, Optional

from pandas import Series

from ..common import Problem
from ..constants import (
    ARTEFACTS_DIR,
    LOCAL_DIR,
    METADATA_FILENAME,
    RESULT_FILENAME,
    SOLUTIONS_DIR,
)
from .artefact import ArtefactContainer
from .download_utils import download_and_extract_archive


class Result(Mapping):
    def __init__(self, source: Mapping[str, float]) -> None:
        self._store = dict(source)
        super().__init__()

    def __getitem__(self, k: str) -> float:
        return self._store.get(k)

    def __iter__(self) -> Iterator[str]:
        return self._store.__iter__()

    def __len__(self) -> int:
        return len(self._store)

    def save(self, path: str) -> None:
        json.dump(self._store, path)

    @staticmethod
    def load(path: str) -> "Result":
        source = json.load(path)
        return Result(source)

    def __repr__(self) -> str:
        return Series(self).__repr__()


class Solution(ArtefactContainer):

    container_dir = SOLUTIONS_DIR

    def __init__(
        self,
        scenario: "Problem",
        objects: Mapping[str, Any],
        id: Optional[str] = None,
        artefacts_url: Optional[str] = None,
        name: Optional[str] = None,
        paper: Optional[str] = None,
        code: Optional[str] = None,
        result: Optional["Result"] = None,
        **kwargs
    ) -> None:

        self.id = id or "".join(
            random.choice(string.ascii_lowercase + string.digits) for _ in range(10)
        )
        self.scenario = scenario
        self.artefacts_url = artefacts_url
        self.name = name
        self.paper = paper
        self.code = code
        self.result = result

        # artefacts = dict(
        #     (
        #         artefact.id,
        #         Artefact(
        #             artefact, container=self, object=objects.get(artefact.id, None)
        #         ),
        #     )
        #     for artefact in scenario.solution_artefacts
        # )
        # self.artefacts = ArtefactBundle(
        #     artefacts, self, [artefacts_url] if artefacts_url is not None else []
        # )

    @property
    def location(self) -> str:
        return os.path.join(LOCAL_DIR, SOLUTIONS_DIR, self.scenario.id, self.id)

    def evaluate(self) -> "Solution":
        self.result = self.scenario.evaluate(self)
        return self

    @staticmethod
    def list(scenario: "Problem") -> List[str]:
        # Determine location of solutions for a specific scenario.
        basedir = os.path.join(LOCAL_DIR, SOLUTIONS_DIR, scenario.id)
        # All child directories correspond to solution ID.
        return [f.name for f in os.scandir(basedir) if f.is_dir()]

    @staticmethod
    def load(scenario: "Problem", id: str) -> "Solution":
        # Determine location of this solution.
        basedir = os.path.join(LOCAL_DIR, SOLUTIONS_DIR, scenario.id, id)

        # Load metadata.
        metadata_path = os.path.join(basedir, METADATA_FILENAME)
        metadata = json.load(metadata_path) if os.path.exists(metadata_path) else dict()
        artefacts_url = metadata.get("artefacts_url", None)
        name = metadata.get("name", None)
        paper = metadata.get("paper", None)
        code = metadata.get("code", None)

        if artefacts_url is not None:
            artefacts_dir = os.path.join(basedir, ARTEFACTS_DIR)
            os.makedirs(artefacts_dir, exist_ok=True)
            download_and_extract_archive(
                artefacts_url, artefacts_dir, remove_finished=True
            )

        # Load result if present.
        result_path = os.path.join(basedir, RESULT_FILENAME)
        result = Result.load(result_path) if os.path.exists(result_path) else None

        return Solution(
            scenario,
            id=id,
            artefacts_url=artefacts_url,
            name=name,
            paper=paper,
            code=code,
            result=result,
        )

    def save(self) -> None:
        # Determine location of this solution.
        basedir = os.path.join(LOCAL_DIR, SOLUTIONS_DIR, self.scenario.id, self.id)
        os.makedirs(basedir, exist_ok=True)

        # Save metadata.
        metadata = dict()
        if self.artefacts_url is not None:
            metadata["artefacts_url"] = self.artefacts_url
        if self.name is not None:
            metadata["name"] = self.name
        if self.paper is not None:
            metadata["paper"] = self.paper
        if self.code is not None:
            metadata["code"] = self.code
        json.dump(metadata, os.path.join(basedir, METADATA_FILENAME))

        # Save artefacts.
        os.makedirs(os.path.join(basedir, ARTEFACTS_DIR), exist_ok=True)
        for artefact in self.artefacts.values():
            artefact.save()

        # Save result.
        if self.result is not None:
            result_path = os.path.join(basedir, RESULT_FILENAME)
            self.result.save(result_path)
