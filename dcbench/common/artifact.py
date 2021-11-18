from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union
from urllib.error import HTTPError
from urllib.request import urlopen, urlretrieve

import meerkat as mk
import pandas as pd
import yaml
from meerkat.tools.lazy_loader import LazyLoader
from torch._C import Value

import dcbench.constants as constants
from dcbench.common.modeling import Model
from dcbench.config import config

from .table import RowMixin

storage = LazyLoader("google.cloud.storage")
torch = LazyLoader("torch")


def _upload_dir_to_gcs(local_path: str, gcs_path: str, bucket: "storage.Bucket"):
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


class Artifact(ABC):

    DEFAULT_EXT: str = ""
    isdir: bool = False

    def __init__(self, artifact_id: str, **kwargs) -> None:
        self.path = f"{artifact_id}.{self.DEFAULT_EXT}"
        self.id = artifact_id
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        super().__init__()

    @property
    def local_path(self) -> str:
        return os.path.join(config.local_dir, self.path)

    @property
    def remote_url(self) -> str:
        return os.path.join(
            config.public_remote_url, self.path + (".tar.gz" if self.isdir else "")
        )

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_path)

    @property
    def is_uploaded(self) -> bool:
        return _url_exists(self.remote_url)

    def upload(self, force: bool = False, bucket: "storage.Bucket" = None):
        if not os.path.exists(self.local_path):
            raise ValueError(
                f"Could not find Artifact to upload at '{self.local_path}'. "
                "Are you sure it is stored locally?"
            )
        if self.is_uploaded and not force:
            return

        if bucket is None:
            client = storage.Client()
            bucket = client.get_bucket(config.public_bucket_name)

        if self.isdir:
            _upload_dir_to_gcs(
                local_path=self.local_path,
                bucket=bucket,
                gcs_path=self.path,
            )
        else:
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
                "Cannot load Artifact that has not been downloaded. "
                "Call `artifact.download()`."
            )

    @abstractmethod
    def load(self) -> Any:
        pass

    @abstractmethod
    def save(self, data: Any) -> None:
        pass

    @classmethod
    def from_data(cls, data: Any, artifact_id: str = None):
        if artifact_id is None:
            artifact_id = uuid.uuid4().hex
        # TODO ():At some point we should probably enforce that ids are unique

        if cls is Artifact:
            # if called on base class, infer which class to use
            if isinstance(data, mk.DataPanel):
                cls = DataPanelArtifact
            elif isinstance(data, pd.DataFrame):
                cls = CSVArtifact
            elif isinstance(data, Model):
                cls = ModelArtifact
            else:
                raise ValueError(
                    f"No Artifact in dcbench for object of type {type(data)}"
                )

        artifact = cls(artifact_id=artifact_id)
        artifact.save(data)
        return artifact

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        data = loader.construct_mapping(node, deep=True)
        return data["class"](artifact_id=data["artifact_id"])

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: Artifact):
        data = {
            "artifact_id": data.id,
            "class": type(data),
        }
        node = dumper.represent_mapping("!Artifact", data)
        return node


# need to use multi_representer to support
yaml.add_multi_representer(Artifact, Artifact.to_yaml)
yaml.add_constructor("!Artifact", Artifact.from_yaml)


class CSVArtifact(Artifact):

    DEFAULT_EXT: str = "csv"

    def load(self) -> pd.DataFrame:
        self._ensure_downloaded()
        data = pd.read_csv(self.local_path, index_col=0)

        def parselists(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except ValueError:
                    return x
            else:
                return x

        return data.applymap(parselists)

    def save(self, data: pd.DataFrame) -> None:
        return data.to_csv(self.local_path)


class YAMLArtifact(Artifact):

    DEFAULT_EXT: str = "yaml"

    def load(self) -> pd.DataFrame:
        self._ensure_downloaded()
        return yaml.load(open(self.local_path), yaml=yaml.FullLoader)

    def save(self, data: Any) -> None:
        return yaml.dump(data, open(self.local_path))


class DataPanelArtifact(Artifact):

    DEFAULT_EXT: str = "mk"
    isdir: bool = True

    def load(self) -> pd.DataFrame:
        self._ensure_downloaded()
        return mk.DataPanel.read(self.local_path)

    def save(self, data: mk.DataPanel) -> None:
        return data.write(self.local_path)


class VisionDatasetArtifact(DataPanelArtifact):

    DEFAULT_EXT: str = "mk"
    isdir: bool = True

    COLUMN_SUBSETS = {
        "celeba": ["id", "image", "identity", "split"],
        "imagenet": ["id", "image", "name", "synset"],
    }

    @classmethod
    def from_name(cls, name: str):
        if name == "celeba":
            dp = mk.datasets.get(name, dataset_dir=config.celeba_dir)
        elif name == "imagenet":
            dp = mk.datasets.get(name, dataset_dir=config.imagenet_dir)
        else:
            raise ValueError(f"No dataset named '{name}' supported by dcbench.")
        dp["id"] = dp["image_id"]
        dp.remove_column("image_id")
        dp = dp[cls.COLUMN_SUBSETS[name]]
        artifact = cls.from_data(data=dp, artifact_id=name)
        return artifact

    def download(self, force: bool = False):
        if self.id == "celeba":
            dp = mk.datasets.get(self.id, dataset_dir=config.celeba_dir)
        elif self.id == "imagenet":
            dp = mk.datasets.get(self.id, dataset_dir=config.imagenet_dir)
        else:
            raise ValueError(f"No dataset named '{self.id}' supported by dcbench.")

        dp["id"] = dp["image_id"]
        dp.remove_column("image_id")
        dp = dp[self.COLUMN_SUBSETS[self.id]]
        self.save(data=dp[self.COLUMN_SUBSETS[self.id]])


class ModelArtifact(Artifact):

    DEFAULT_EXT: str = "pt"

    def load(self) -> Model:
        self._ensure_downloaded()
        dct = torch.load(self.local_path, map_location="cpu")
        model = dct["class"](dct["config"])
        model.load_state_dict(dct["state_dict"])
        return model

    def save(self, data: Model) -> None:
        return torch.save(
            {
                "state_dict": data.state_dict(),
                "config": data.config,
                "class": type(data),
            },
            self.local_path,
        )


BASIC_TYPE = Union[int, float, str, bool]


@dataclass
class ArtifactSpec:
    description: str
    artifact_type: type


class ArtifactContainer(ABC, Mapping, RowMixin):

    artifact_specs: Mapping[str, ArtifactSpec]
    task_id: str = "none"
    container_type: str

    def __init__(
        self,
        id: str,
        artifacts: Mapping[str, Artifact],
        attributes: Mapping[str, BASIC_TYPE] = None,
    ):
        super().__init__(id=id)
        artifacts = self._create_artifacts(artifacts=artifacts)
        self._check_artifact_specs(artifacts=artifacts)
        self.artifacts = artifacts
        if attributes is None:
            attributes = {}
        self._attributes = attributes

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Artifact],
        attributes: Mapping[str, BASIC_TYPE] = None,
        container_id: str = None,
    ):
        if container_id is None:
            container_id = uuid.uuid4().hex
        container = cls(id=container_id, artifacts=artifacts, attributes=attributes)
        return container

    def __getitem__(self, key):
        artifact = self.artifacts.__getitem__(key)
        if not artifact.is_downloaded:
            artifact.download()
        return self.artifacts.__getitem__(key).load()

    def __iter__(self):
        return self.artifacts.__iter__()

    def __len__(self):
        return self.artifacts.__len__()

    def __getattr__(self, k: str) -> Any:
        try:
            return self.attributes[k]
        except KeyError:
            raise AttributeError(k)

    @property
    def is_downloaded(self) -> bool:
        return all(x.is_downloaded for x in self.artifacts.values())

    @property
    def is_uploaded(self) -> bool:
        return all(x.is_uploaded for x in self.artifacts.values())

    def upload(self, force: bool = False, bucket: "storage.Bucket" = None):
        if bucket is None:
            client = storage.Client()
            bucket = client.get_bucket(config.public_bucket_name)

        for artifact in self.artifacts.values():
            artifact.upload(force=force, bucket=bucket)

    def download(self, force: bool = False) -> bool:
        for artifact in self.artifacts.values():
            artifact.download(force=force)

    def _create_artifacts(self, artifacts: Mapping[str, Artifact]):
        return {
            name: artifact
            if isinstance(artifact, Artifact)
            else Artifact.from_data(
                data=artifact,
                artifact_id=os.path.join(
                    self.task_id,
                    self.container_type,
                    constants.ARTIFACTS_DIR,
                    self.id,
                    name,
                ),
            )
            for name, artifact in artifacts.items()
        }

    @classmethod
    def _check_artifact_specs(cls, artifacts: Mapping[str, Artifact]):
        for name, artifact in artifacts.items():
            if name not in cls.artifact_specs:
                raise ValueError(
                    f"Passed artifact name '{name}', but the specification for"
                    f" {cls.__name__} doesn't include it."
                )

            if not isinstance(artifact, cls.artifact_specs[name].artifact_type):
                raise ValueError(
                    f"Passed an artifact of type {type(artifact)} to {cls.__name__}"
                    f" for the artifact named '{name}'. The specification for"
                    f" {cls.__name__} expects an Artifact of type"
                    f" {cls.artifact_specs[name].artifact_type}."
                )

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        data = loader.construct_mapping(node, deep=True)
        return data["class"](
            id=data["container_id"],
            artifacts=data["artifacts"],
            attributes=data["attributes"],
        )

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: ArtifactContainer):
        data = {
            "class": type(data),
            "container_id": data.id,
            "attributes": data._attributes,
            "artifacts": data.artifacts,
        }
        return dumper.represent_mapping("!ArtifactContainer", data)

    def __repr__(self):
        artifacts = {k: v.__class__.__name__ for k, v in self.artifacts.items()}
        return (
            f"{self.__class__.__name__}(artifacts={artifacts}, "
            f"attributes={self.attributes})"
        )


yaml.add_multi_representer(ArtifactContainer, ArtifactContainer.to_yaml)
yaml.add_constructor("!ArtifactContainer", ArtifactContainer.from_yaml)
