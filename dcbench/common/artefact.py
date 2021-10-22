import os
import pandas as pd

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Optional, Type, Iterator, List, Union

from pandas.core.frame import DataFrame

from dcbench.constants import ARTEFACTS_DIR

from .bundle import RelationalBundle, Bundle
from ..constants import HIDDEN_ARTEFACTS_URL
from .download_utils import download_and_extract_archive


class ArtefactContainer(ABC):

    @property
    @abstractmethod
    def location() -> str:
        pass


class Artefact(ABC):

    DEFAULT_EXT : str = ""

    def __init__(self, id: str, filename: Optional[str] = None, optional: bool = False, **kwargs) -> None:

        self.id = id
        if filename is None and self.DEFAULT_EXT:
            filename = id + "." + self.DEFAULT_EXT
        self.filename = filename
        self.optional = optional

    def location(self, basepath: str) -> str:
        return os.path.join(basepath, ARTEFACTS_DIR, self.filename)

    def downloaded(self, basepath: str) -> bool:
        return os.path.exists(self.location(basepath))
    
    @abstractmethod
    def load(self, basepath: str) -> Any:
        pass
    
    @abstractmethod
    def save(self, basepath: str) -> None:
        pass


class CsvArtefact(Artefact):

    DEFAULT_EXT : str = "csv"

    def __init__(self, id: str, filename: Optional[str] = None, optional: bool = False, **kwargs) -> None:
        self.object : Optional[DataFrame] = None
        super().__init__(id, filename, optional, **kwargs)
    
    def load(self, basepath: str) -> Any:
        if self.object is None:
            self.object = pd.read_csv(self.location(basepath))
        return self.object

    def save(self, basepath: str) -> None:
        return self.object.to_csv(self.location(basepath))


class ArtefactInstance:

    def __init__(self, artefact: Artefact, container: ArtefactContainer, object: Optional[Any] = None) -> None:
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

    def __init__(self, artefacts: List[ArtefactInstance], container: ArtefactContainer, url: Optional[Union[str, List[str]]], **kwargs) -> None:
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
        return all(x.downloaded or (HIDDEN_ARTEFACTS_URL is None and x.artefact.optional) for x in self.values())

    def load(self) -> Bundle[Any]:
        if not self.downloaded:
            self.download()

        return Bundle[Any](dict((k, v.load()) for (k, v) in self.items() if v.downloaded))
