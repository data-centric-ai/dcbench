from __future__ import annotations
import os
import pandas as pd
import meerkat as mk

import uuid
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError
import yaml
import tempfile
import subprocess

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Optional, Type, Iterator, List, Union

from meerkat.tools.lazy_loader import LazyLoader

from dcbench.constants import ARTEFACTS_DIR, BUCKET_NAME, LOCAL_DIR, PUBLIC_REMOTE_URL

storage = LazyLoader("google.cloud.storage")
torch = LazyLoader("torch")


def _upload_dir_to_gcs(local_path: str, bucket_name: str, gcs_path: str):

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    assert os.path.isdir(local_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tarball_path = os.path.join(tmp_dir, "run.tar.gz")
        subprocess.call(
            [
                "tar",
                "-czf",
                tarball_path,
                "-C",
                local_path,
                ".",
            ]
        )
        remote_path = gcs_path + ".tar.gz"
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(tarball_path)


def _url_exists(url: str):
    try:
        response = urlopen(url)
        status_code = response.getcode()
        return status_code == 200
    except HTTPError:
        return False


"""[summary]

artefacts
    common
    slice
    miniclean

"""
mk.datasets.get("imagenet")

class Artefact(ABC):

    DEFAULT_EXT: str = ""
    isdir: bool = False

    def __init__(self, artefact_id: str, **kwargs) -> None:
        self.path = os.path.join(ARTEFACTS_DIR, f"{artefact_id}.{self.DEFAULT_EXT}")
        self.id = artefact_id
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        super().__init__()

    @property
    def local_path(self) -> str:
        return os.path.join(LOCAL_DIR, self.path)

    @property
    def remote_url(self) -> str:
        return os.path.join(
            PUBLIC_REMOTE_URL, self.path + ".tar.gz" if self.isdir else ""
        )

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_path)

    @property
    def is_uploaded(self) -> bool:
        return _url_exists(self.remote_url)

    def upload(self, force: bool = False):
        if not os.path.exists(self.local_path):
            raise ValueError(
                "Could not find Artefact to upload. "
                "Are you sure it is stored locally?"
            )
        if self.is_uploaded and not force:
            return 
            
        if self.isdir:
            _upload_dir_to_gcs(
                local_path=self.local_path, bucket_name=BUCKET_NAME, gcs_path=self.path
            )
        else:
            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(self.path)
            blob.upload_from_filename(self.local_path)

    def download(self, force: bool = False):
        if self.is_downloaded and not force:
            return

        if self.isdir:
            os.makedirs(self.local_path, exist_ok=True)
            tarball_path = self.local_path + ".tar.gz"
            urlretrieve(self.remote_url, tarball_path)
            subprocess.call(["tar", "-xzf", tarball_path, "-C", self.local_path])

        else:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            urlretrieve(self.remote_url, self.local_path)

    def _ensure_downloaded(self):
        if not self.is_downloaded:
            raise ValueError(
                "Cannot load Artefact that has not been downloaded."
                "Call `artefact.download()`."
            )

    @abstractmethod
    def load(self) -> Any:
        pass

    @abstractmethod
    def save(self, data: any) -> None:
        pass

    @classmethod
    def from_data(cls, data: any, artefact_id: str = None):
        if artefact_id is None:
            artefact_id = uuid.uuid4().hex

        # TODO ():At some point we should probably enforce that ids are unique
        artefact = cls(artefact_id=artefact_id)
        artefact.save(data)
        return artefact

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        data = loader.construct_mapping(node)
        return data["class"](artefact_id=data["artefact_id"])

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: Artefact):
        data = {
            "artefact_id": data.id,
            "class": type(data),
        }
        node = dumper.represent_mapping("!Artefact", data)
        return node


# need to use multi_representer to support
yaml.add_multi_representer(Artefact, Artefact.to_yaml)
yaml.add_constructor("!Artefact", Artefact.from_yaml)


class CSVArtefact(Artefact):

    DEFAULT_EXT: str = "csv"

    def load(self) -> pd.DataFrame:
        self._ensure_downloaded()
        return pd.read_csv(self.local_path)

    def save(self, data: pd.DataFrame) -> None:
        return data.to_csv(self.local_path)


class DataPanelArtefact(Artefact):

    DEFAULT_EXT: str = "mk"
    isdir: bool = True

    def load(self) -> pd.DataFrame:
        self._ensure_downloaded()
        return mk.DataPanel.read(self.local_path)

    def save(self, data: mk.DataPanel) -> None:
        return data.write(self.local_path)


class ModelArtefact(Artefact):

    DEFAULT_EXT: str = "pt"

    def load(self) -> pd.DataFrame:
        pass
        # TODO: add custom model class

    def save(self, data) -> None:
        return torch.save({"state_dict": data.state_dict()}, self.local_path)

class ImageNetDatasetArtefact

BASIC_TYPE = Union[int, float, str, bool]


class ArtefactContainer(ABC, Mapping):

    artefact_spec: Mapping[str, type]
    container_dir: str
    task_id: str = "none"

    def __init__(
        self,
        container_id: str,
        artefacts: Mapping[str, Artefact],
        attributes: Mapping[str, BASIC_TYPE] = None,
    ):
        self._check_artefact_spec(artefacts=artefacts)
        self.artefacts = artefacts
        self.container_id = container_id
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    @classmethod
    def from_artefacts(
        cls,
        artefacts: Mapping[str, Artefact],
        attributes: Mapping[str, BASIC_TYPE] = None,
    ):
        container_id = uuid.uuid4().hex
        container = cls(
            container_id=container_id, artefacts=artefacts, attributes=attributes
        )
        return container

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = value

    def __getitem__(self, key):
        return self.artefacts.__getitem__(key).load()

    def __iter__(self):
        return self.artefacts.__iter__()

    def __len__(self):
        return self.artefacts.__len__()

    @property
    def is_downloaded(self) -> bool:
        return all(x.is_downloaded for x in self.artefacts.values())

    @property
    def is_uploaded(self) -> bool:
        return all(x.is_uploaded for x in self.artefacts.values())

    def upload(self):
        for artefact in self.artefacts.values():
            artefact.upload()

    def download(self, force: bool = False) -> bool:
        for artefact in self.artefacts.values():
            artefact.download(force=force)

    @classmethod
    def create_set(cls, containers: List[ArtefactContainer]):
        for container in containers:
            assert isinstance(container, cls)
            container.upload()
        
        yaml.dump(containers, open("/tasks/slice_discovery/problems.yaml"))
        upload("/tasks")
        pass 
        

    @classmethod
    def list(cls):
        # do the loading here, not in the init 
        cls = yaml.load(containers, open("/tasks/slice_discovery/problems.yaml"))
        returns cls.problems # maybe a dataframe 

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
            if name not in cls.artefact_spec:
                raise ValueError(
                    f"Passed artefact name '{name}', but the specification for"
                    f" {cls.__name__} doesn't include it."
                )

            if not isinstance(artefact, cls.artefact_spec[name]):
                raise ValueError(
                    f"Passed an artefact of type {type(artefact)} to {cls.__name__}"
                    f" for the artefact named '{name}'. The specification for"
                    f" {cls.__name__} expects an Artefact of type"
                    f" {cls.artefact_spec[name]}."
                )

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        data = loader.construct_mapping(node)
        return data["class"](
            container_id=data["container_id"],
            artefacts=data["artefacts"],
            attributes=data["attributes"],
        )

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: ArtefactContainer):
        data = {
            "class": type(data),
            "container_id": data.container_id,
            "attributes": data.attributes,
            "artefacts": data.artefacts,
        }
        return dumper.represent_mapping("!ArtefactContainer", data)


yaml.add_multi_representer(ArtefactContainer, ArtefactContainer.to_yaml)
yaml.add_constructor("!ArtefactContainer", ArtefactContainer.from_yaml)
