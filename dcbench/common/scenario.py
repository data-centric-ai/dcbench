import json
import os
import random
import string
import urllib.parse

from abc import abstractmethod
from pandas import Series
from typing import Mapping, Optional, List, Any, Iterator, Type

from dcbench.common.bundle import RelationalBundle

from .artefact import Artefact, ArtefactContainer, ArtefactInstance, ArtefactBundle
from ..constants import DEFAULT_WORKING_DIR, METADATA_FILENAME, RESULT_FILENAME, SCENARIOS_DIR, SOLUTIONS_DIR, ARTEFACTS_DIR, PUBLIC_ARTEFACTS_URL, HIDDEN_ARTEFACTS_URL
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

    def __init__(self, scenario: "Scenario", objects: Mapping[str, Any],
        id: Optional[str] = None, artefacts_url: Optional[str] = None,
        name: Optional[str] = None, paper: Optional[str] = None, code: Optional[str] = None,
        result: Optional["Result"] = None, **kwargs) -> None:

        self.id = id or "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        self.scenario = scenario
        self.artefacts_url = artefacts_url
        self.name = name
        self.paper = paper
        self.code = code
        self.result = result

        artefacts = dict((artefact.id, ArtefactInstance(artefact, container=self, object=objects.get(artefact.id, None))) for artefact in scenario.solution_artefacts)
        self.artefacts = ArtefactBundle(artefacts, self, [artefacts_url] if artefacts_url is not None else [])

    @property
    def location(self) -> str:
        return os.path.join(DEFAULT_WORKING_DIR, SOLUTIONS_DIR, self.scenario.id, self.id)
    
    def evaluate(self) -> "Solution":
        self.result = self.scenario.evaluate(self)
        return self
    
    @staticmethod
    def list(scenario: "Scenario") -> List[str]:
        # Determine location of solutions for a specific scenario.
        basedir = os.path.join(DEFAULT_WORKING_DIR, SOLUTIONS_DIR, scenario.id)
        # All child directories correspond to solution ID.
        return [f.name for f in os.scandir(basedir) if f.is_dir()]
    
    @staticmethod
    def load(scenario: "Scenario", id: str) -> "Solution":
        # Determine location of this solution.
        basedir = os.path.join(DEFAULT_WORKING_DIR, SOLUTIONS_DIR, scenario.id, id)

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
            download_and_extract_archive(artefacts_url, artefacts_dir, remove_finished=True)

        # Load result if present.
        result_path = os.path.join(basedir, RESULT_FILENAME)
        result = Result.load(result_path) if os.path.exists(result_path) else None

        return Solution(scenario, id=id, artefacts_url=artefacts_url, name=name, paper=paper, code=code, result=result)


    def save(self) -> None:
        # Determine location of this solution.
        basedir = os.path.join(DEFAULT_WORKING_DIR, SOLUTIONS_DIR, self.scenario.id, self.id)
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


class Scenario(ArtefactContainer):

    scenarios : Mapping[str, Type["Scenario"]] = {}
    id : Optional[str] = None
    scenario_artefacts : Optional[List[Artefact]] = None
    solution_artefacts : Optional[List[Artefact]] = None
    result_metrics : Optional[List[str]] = None

    def __init_subclass__(cls, id: Optional[str] = None, scenario_artefacts: Optional[List[Artefact]] = None,
        solution_artefacts: Optional[List[Artefact]] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.id = id
        if id is not None:
            cls.scenarios[id] = cls
        if scenario_artefacts is not None:
            cls.scenario_artefacts = scenario_artefacts
        if solution_artefacts is not None:
            cls.solution_artefacts = solution_artefacts

    def __init__(self, working_dir: Optional[str] = None) -> None:
        self.working_dir : str = working_dir or DEFAULT_WORKING_DIR
        self._artefacts : Optional[ArtefactBundle] = None
        self._solutions : Optional[List[Solution]] = None

    @property
    def location(self) -> str:
        return os.path.join(self.working_dir, SCENARIOS_DIR, self.id)

    @property
    def downloaded(self) -> bool:
        return all(x.downloaded or (HIDDEN_ARTEFACTS_URL is None and x.artefact.optional) for x in self.artefacts.values())

    def download(self) -> None:
        if self.id is None:
            raise TypeError("Can not download data for a scenario without an id.")

        os.makedirs(self.location, exist_ok=True)

        scenario_public_artefacts_url = urllib.parse.urljoin(PUBLIC_ARTEFACTS_URL, self.id)
        download_and_extract_archive(scenario_public_artefacts_url, self.location, remove_finished=True)

        if HIDDEN_ARTEFACTS_URL is not None:
            scenario_hidden_artefacts_url = urllib.parse.urljoin(HIDDEN_ARTEFACTS_URL, self.id)
            download_and_extract_archive(scenario_hidden_artefacts_url, self.location, remove_finished=True)

    @property
    def solutions(self) -> RelationalBundle[Solution]:
        if self._solutions is None:
            solutions = dict((id, Solution.load(self, id)) for id in Solution.list(self))
            all_result_attributes = set.union(*[s.result.keys() for s in solutions.values() if s.result is not None])
            attributes = ["name", "paper", "code"] + ["result.%s" % a for a in sorted(all_result_attributes)]
            self._solutions = RelationalBundle(solutions, attributes)
        return self._solutions

    @classmethod
    def list(cls) -> List[str]:
        return cls.scenarios.keys()

    @property
    def artefacts(self) -> ArtefactBundle:
        
        if self.scenario_artefacts is None:
            raise NotImplementedError("Each scenario class must have a defined collection of scenario_artefacts.")

        if self._artefacts is None:
            url = [urllib.parse.urljoin(PUBLIC_ARTEFACTS_URL, self.id)]
            if HIDDEN_ARTEFACTS_URL is not None:
                url += [urllib.parse.urljoin(HIDDEN_ARTEFACTS_URL, self.id)]
            artefacts = dict((artefact.id, ArtefactInstance(artefact, self)) for artefact in self.scenario_artefacts)
            self._artefacts = ArtefactBundle(artefacts, self, url)
        return self._artefacts

    @abstractmethod
    def solve(self, **kwargs: Any) -> Solution:
        pass

    @abstractmethod
    def evaluate(self, solution: Solution) -> Result:
        pass
