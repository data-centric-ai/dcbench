from __future__ import annotations
import os
import pandas as pd
import uuid
from urllib.request import urlretrieve
from functools import lru_cache
import yaml

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Optional, Type, Iterator, List, Union

from pandas.core.frame import DataFrame

from dcbench.constants import ARTEFACTS_DIR, BUCKET_NAME, LOCAL_DIR, PUBLIC_REMOTE_URL

from .bundle import RelationalBundle, Bundle
from .download_utils import download_and_extract_archive


class Artefact(ABC):

    DEFAULT_EXT: str = ""

    def __init__(self, artefact_id: str, task_id: str, **kwargs) -> None:
        self.path = os.path.join(
            task_id, ARTEFACTS_DIR, f"{artefact_id}.{self.DEFAULT_EXT}"
        )
        self.id = artefact_id
        self.task_id = task_id
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)

    @property
    def local_path(self) -> str:
        return os.path.join(LOCAL_DIR, self.path)

    @property
    def remote_url(self) -> str:
        return os.path.join(PUBLIC_REMOTE_URL, self.path)

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_path)

    @property
    def is_uploaded(self) -> bool:
        return os.path.exists(self.local_path)

    def upload(self):
        import google.cloud.storage as storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(self.path)
        blob.upload_from_filename(self.local_path)

    def download(self):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        urlretrieve(self.remote_url, self.local_path)

    @abstractmethod
    def load(self) -> Any:
        pass

    @abstractmethod
    def save(self, data: any) -> None:
        pass

    @classmethod
    def from_data(cls, data: any, task_id: str, artefact_id: str = None):
        if artefact_id is None:
            artefact_id = uuid.uuid4().hex

        # TODO ():At some point we should probably enforce that ids are unique
        artefact = cls(artefact_id=artefact_id, task_id=task_id)
        artefact.save(data)
        return artefact


class CSVArtefact(Artefact):

    DEFAULT_EXT: str = "csv"

    def load(self) -> Any:
        if self.object is None:
            self.object = pd.read_csv(self.local_path)
        return self.object

    def save(self, data: pd.DataFrame) -> None:
        return data.to_csv(self.local_path)




class ArtefactContainer(ABC, Mapping):

    artefact_spec: Mapping[str, type]
    container_dir: str
    task_id: str = "none"

    def __init__(self, container_id: str, artefacts: Mapping[str, Artefact]):
        self._check_artefact_spec(artefacts=artefacts)
        self.artefacts = artefacts
        self.path = os.path.join(
            self.task_id, self.container_dir, f"{container_id}.yaml"
        )
        self.container_id = container_id

    @classmethod
    def from_artefacts(
        cls, artefacts: Mapping[str, Artefact], container_id: str = None
    ):
        if container_id is None:
            container_id = uuid.uuid4().hex
        container = cls(container_id=uuid.uuid4().hex, artefacts=artefacts)
        container.save()
        return container

    @property
    def attributes(self):
        return {}

    @attributes.setter
    def attributes(self, value):
        pass

    def __getitem__(self, key):
        return self.artefacts.__getitem__(key).load()

    def __iter__(self):
        return self.artefacts.__iter__()

    def __len__(self):
        return self.artefacts.__len__()

    @property
    def local_path(self) -> str:
        return os.path.join(LOCAL_DIR, self.path)

    @property
    def remote_url(self) -> str:
        return os.path.join(PUBLIC_REMOTE_URL, self.path)

    @property
    def is_downloaded(self) -> bool:
        return all(x.downloaded for x in self.artefacts.values())

    @property
    def is_uploaded(self) -> bool:
        return os.path.exists(self.local_path)

    def upload(self):
        import google.cloud.storage as storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(self.path)
        blob.upload_from_filename(self.local_path)

    def save(self):
        data = {
            "container_id": self.id,
            "attributes": self.attributes,
            "artefacts": {
                name: {
                    "artefact_id": artefact.id,
                    "task_id": artefact.task_id,
                    "class": type(artefact),
                }
                for name, artefact in self.artefacts.items()
            },
        }
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        yaml.dump(data, open(self.local_path, "w"))

    @classmethod
    def from_id(cls, container_id: str):
        data = yaml.load(open(path, "r"))
        artefacts = {
            name: a["class"](artefact_id=a["id"], task_id=a["task_id"])
            for name, a in data["artefacts"].items()
        }
        container = cls(
            container_id=container_id,
            artefacts=artefacts,
        )
        container.attributes = data["attributes"]
        return container

    @classmethod
    def _check_artefact_spec(cls, artefacts: Mapping[str, Artefact]):
        for name, artefact in artefacts.items():
            if not isinstance(artefact, cls.artefact_spec[name]):
                raise ValueError(
                    f"Passed an artefact of type {type(artefact)} to {cls.__name__}"
                    f" for the artefact named '{name}'. The specification for"
                    f" {cls.__name__} expects an Artefact of type"
                    f" {cls.artefact_spec[name]}."
                )
