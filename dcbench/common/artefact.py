from __future__ import annotations
import os
import pandas as pd
from urllib.request import urlretrieve

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Optional, Type, Iterator, List, Union

from pandas.core.frame import DataFrame

from dcbench.constants import ARTEFACTS_DIR, LOCAL_DIR, PUBLIC_REMOTE_URL

from .bundle import RelationalBundle, Bundle
from .download_utils import download_and_extract_archive


class Artefact(ABC):

    DEFAULT_EXT: str = ""

    def __init__(
        self, artefact_id: str,  **kwargs
    ) -> None:
        self.path = os.path.join(ARTEFACTS_DIR, f"{artefact_id}.{self.DEFAULT_EXT}")
        self.id = artefact_id

    @property
    def local_path(self) -> str:
        return os.path.join(LOCAL_DIR, self.path)
    @property
    def remote_url(self) -> str:
        return os.path.join(PUBLIC_REMOTE_URL, self.path)
    
    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_path)

    def download(self): 
        urlretrieve(self.remote_url, self.local_path) 
        
    @abstractmethod
    def load(self, basepath: str) -> Any:
        pass

    @abstractmethod
    def save(self, basepath: str) -> None:
        pass


class CSVArtefact(Artefact):

    DEFAULT_EXT: str = "csv"

    def __init__(
        self, artefact_id: str, **kwargs
    ) -> None:
        self.object: Optional[DataFrame] = None
        super().__init__(artefact_id=artefact_id, **kwargs)

    def load(self, basepath: str) -> Any:
        if self.object is None:
            self.object = pd.read_csv(self.location(basepath))
        return self.object

    def save(self, basepath: str) -> None:
        return self.object.to_csv(self.location(basepath))


class DataPanelArtefact(Artefact):

    DEFAULT_EXT: str = "mk"

    def __init__(self):
        pass


class ModelArtefact(Artefact):

    DEFAULT_EXT: str = "pt"

    def __init__(self):
        pass


class ArtefactContainer(ABC, Mapping):

    artefact_spec: Mapping[str, type]

    def __init__(self, artefacts: Mapping[str, Artefact]):
        self._check_artefact_spec(artefacts=artefacts)
        self.artefacts = artefacts

    @property
    def artefacts(self):

        if self.scenario_artefacts is None:
            raise NotImplementedError(
                "Each scenario class must have a defined collection of scenario_artefacts."
            )

        if self._artefacts is None:
            url = [urllib.parse.urljoin(PUBLIC_ARTEFACTS_URL, self.id)]
            if HIDDEN_ARTEFACTS_URL is not None:
                url += [urllib.parse.urljoin(HIDDEN_ARTEFACTS_URL, self.id)]
            artefacts = dict(
                (artefact.id, ArtefactInstance(artefact, self))
                for artefact in self.scenario_artefacts
            )
            self._artefacts = ArtefactBundle(artefacts, self, url)
        return self._artefacts

    def __getitem__(self, key):
        return self.artefacts.__getitem__(key).load()

    def __iter__(self):
        return self.artefacts.__iter__()

    def __len__(self):
        return self.artefacts.__len__()

    @property
    def downloaded(self) -> bool:
        return all(
            x.downloaded or (HIDDEN_ARTEFACTS_URL is None and x.artefact.optional)
            for x in self.artefacts.values()
        )

    def download(self) -> None:
        if self.id is None:
            raise TypeError("Can not download data for a scenario without an id.")

        os.makedirs(self.location, exist_ok=True)

        scenario_public_artefacts_url = urllib.parse.urljoin(
            PUBLIC_ARTEFACTS_URL, self.id
        )
        download_and_extract_archive(
            scenario_public_artefacts_url, self.location, remove_finished=True
        )

        if HIDDEN_ARTEFACTS_URL is not None:
            scenario_hidden_artefacts_url = urllib.parse.urljoin(
                HIDDEN_ARTEFACTS_URL, self.id
            )
            download_and_extract_archive(
                scenario_hidden_artefacts_url, self.location, remove_finished=True
            )

    def upload():
        pass

    @classmethod
    def _check_artefact_spec(cls, artefacts: Mapping[str, Artefact]):
        for name, artefact in artefacts.items():
            if not isinstance(artefact, cls.artefact_spec[name]):
                raise ValueError(
                    f"Passed an artefact of type {type(artefact)} to {cls.__name__ƒ}"
                    f" for the artefact named '{name}'. The specification for"
                    f" {cls.__name__} expects an Artefact of type"
                    f" {cls.artefact_spec[name]}."
                )


class ArtefactInstance:
    def __init__(
        self,
        artefact: Artefact,
        container: ArtefactContainer,
        object: Optional[Any] = None,
    ) -> None:
        self.artefact = artefact
        self.container = container
        self.object = object

    @property
    def location(self) -> str:
        return self.artefact.location(self.container.location)

    @property
    def downloaded(self) -> bool:
        return self.artefact.downloaded(self.container.location)

    def load(self) -> Any:
        if self.object is None:
            self.object = self.artefact.load(self.container.location)
        return self.object

    def save(self) -> None:
        self.artefact.save(self.container.location)


class ArtefactBundle(RelationalBundle[ArtefactInstance]):
    def __init__(
        self,
        artefacts: List[ArtefactInstance],
        container: ArtefactContainer,
        url: Optional[Union[str, List[str]]],
        **kwargs,
    ) -> None:
        self.container = container
        self.url = url

        super().__init__(artefacts, attributes=["location"], **kwargs)

    @property
    def location(self) -> str:
        return os.path.join(self.container.location, ARTEFACTS_DIR)

    def download(self) -> "ArtefactBundle":
        os.makedirs(self.location, exist_ok=True)
        for url in self.url:
            download_and_extract_archive(url, self.location, remove_finished=True)

        return self

    @property
    def downloaded(self) -> bool:
        return all(
            x.downloaded or (HIDDEN_ARTEFACTS_URL is None and x.artefact.optional)
            for x in self.values()
        )

    def load(self) -> Bundle[Any]:
        if not self.downloaded:
            self.download()

        return Bundle[Any](
            dict((k, v.load()) for (k, v) in self.items() if v.downloaded)
        )
