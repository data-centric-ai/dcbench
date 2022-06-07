from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import Any, Union
from urllib.error import HTTPError
from urllib.request import urlopen, urlretrieve
import warnings

import meerkat as mk
import pandas as pd
import yaml
from meerkat.tools.lazy_loader import LazyLoader

from dcbench.common.modeling import Model
from dcbench.config import config

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

def urlretrieve_with_retry(url: str, filename: str, max_retries: int=5):
    """
    Retry urlretrieve() if it fails.
    """
    for idx in range(max_retries):
        try:
            urlretrieve(url, filename)
            return
        except Exception as e:
            warnings.warn(
                f"Failed to download {url}: {e}\n"
                f"Retrying {idx}/{max_retries}..."
            )
            continue
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries.")


class Artifact(ABC):
    """A pointer to a unit of data (e.g. a CSV file) that is stored locally on
    disk and/or in a remote GCS bucket.

    In DCBench, each artifact is identified by a unique artifact ID. The only
    state that the :class:`Artifact` object must maintain is this ID (``self.id``).
    The object does not hold the actual data in memory, making it
    lightweight.

    :class:`Artifact` is an abstract base class. Different types of artifacts (e.g. a
    CSV file vs. a PyTorch model) have corresponding subclasses of :class:`Artifact`
    (e.g. :class:`CSVArtifact`, :class:`ModelArtifact`).

    .. Tip::
        The vast majority of users should not call the :class:`Artifact`
        constructor directly. Instead, they should either create a new artifact by
        calling :meth:`from_data` or load an existing artifact from a YAML file.

    The class provides utilities for accessing and managing a unit of data:

    - Synchronizing the local and remote copies of a unit of data:
      :meth:`upload`, :meth:`download`
    - Loading the data into memory: :meth:`load`
    - Creating new artifacts from in-memory data: :meth:`from_data`
    - Serializing the pointer artifact so it can be shared:
      :meth:`to_yaml`, :meth:`from_yaml`


    Args:
        artifact_id (str): The unique artifact ID.

    Attributes:
        id (str): The unique artifact ID.
    """

    @classmethod
    def from_data(
        cls, data: Union[mk.DataPanel, pd.DataFrame, Model], artifact_id: str = None
    ) -> Artifact:
        """Create a new artifact object from raw data and save the artifact to
        disk in the local directory specified in the config file at
        ``config.local_dir``.

        .. tip::

            When called on the abstract base class :class:`Artifact`, this method will
            infer which artifact subclass to use. If you know exactly which artifact
            class you'd like to use (e.g. :class:`DataPanelArtifact`), you should call
            this classmethod on that subclass.

        Args:
            data (Union[mk.DataPanel, pd.DataFrame, Model]): The raw data that will be
                saved to disk.
            artifact_id (str, optional): . Defaults to None, in which case a UUID will
                be generated and used.

        Returns:
            Artifact: A new artifact pointing to the :arg:`data` that was saved to disk.
        """
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
            elif isinstance(data, (list, dict)):
                cls = YAMLArtifact
            else:
                raise ValueError(
                    f"No Artifact in dcbench for object of type {type(data)}"
                )

        artifact = cls(artifact_id=artifact_id)
        artifact.save(data)
        return artifact

    @property
    def local_path(self) -> str:
        """The local path to the artifact in the local directory specified in
        the config file at ``config.local_dir``."""
        return os.path.join(config.local_dir, self.path)

    @property
    def remote_url(self) -> str:
        """The URL of the artifact in the remote GCS bucket specified in the
        config file at ``config.public_bucket_name``."""
        return os.path.join(
            config.public_remote_url, self.path + (".tar.gz" if self.isdir else "")
        )

    @property
    def is_downloaded(self) -> bool:
        """Checks if artifact is downloaded to local directory specified in the
        config file at ``config.local_dir``.

        Returns:
            bool: True if artifact is downloaded, False otherwise.
        """
        return os.path.exists(self.local_path)

    @property
    def is_uploaded(self) -> bool:
        """Checks if artifact is uploaded to GCS bucket specified in the config
        file at ``config.public_bucket_name``.

        Returns:
            bool: True if artifact is uploaded, False otherwise.
        """
        return _url_exists(self.remote_url)

    def upload(self, force: bool = False, bucket: "storage.Bucket" = None) -> bool:
        """Uploads artifact to a GCS bucket at ``self.path``, which by default
        is just the artifact ID with the default extension.

        Args:
            force (bool, optional): Force upload even if artifact is already uploaded.
                Defaults to False.
            bucket (storage.Bucket, optional): The GCS bucket to which the artifact is
                uplioaded. Defaults to None, in which case the artifact is uploaded to
                the bucket speciried in the config file at config.public_bucket_name.

        Returns
            bool: True if artifact was uploaded, False otherwise.
        """

        if not os.path.exists(self.local_path):
            raise ValueError(
                f"Could not find Artifact to upload at '{self.local_path}'. "
                "Are you sure it is stored locally?"
            )
        if self.is_uploaded and not force:
            warnings.warn(
                f"Artifact {self.id} is not being re-uploaded."
                "Set `force=True` to force upload."
            )
            return False

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
            blob.metadata = {"Cache-Control": "private, max-age=0, no-transform"}
            blob.patch()
        return True

    def download(self, force: bool = False) -> bool:
        """Downloads artifact from GCS bucket to the local directory specified
        in the config file at ``config.local_dir``. The relative path to the
        artifact within that directory is ``self.path``, which by default is
        just the artifact ID with the default extension.

        Args:
            force (bool, optional): Force download even if artifact is already
             downloaded. Defaults to False.

        Returns:
            bool: True if artifact was downloaded, False otherwise.

        .. warning::
            By default, the GCS cache on public urls has a max-age up to an hour.
            Therefore, when updating an existin artifacts, changes may not be
            immediately reflected in subsequent downloads.

            See `here
            <https://stackoverflow.com/questions/62897641/google-cloud-storage-public-ob
            ject-url-e-super-slow-updating>`_
            for more details.
        """

        if self.is_downloaded and not force:

            return False
        if self.isdir:
            if self.is_downloaded:
                shutil.rmtree(self.local_path)
            os.makedirs(self.local_path, exist_ok=True)
            tarball_path = self.local_path + ".tar.gz"
            urlretrieve_with_retry(self.remote_url, tarball_path)
            subprocess.call(["tar", "-xzf", tarball_path, "-C", self.local_path])
            os.remove(tarball_path)

        else:
            if self.is_downloaded:
                os.remove(self.local_path)
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            urlretrieve_with_retry(self.remote_url, self.local_path)

        return True

    DEFAULT_EXT: str = ""
    isdir: bool = False

    @abstractmethod
    def load(self) -> Any:
        """Load the artifact into memory from disk at ``self.local_path``."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Any) -> None:
        """Save data to disk at ``self.local_path``."""
        raise NotImplementedError()

    def __init__(self, artifact_id: str, **kwargs) -> None:
        """
        .. warning::
            In general, you should not instantiate an Artifact directly. Instead, use
            :meth:`Artifact.from_data` to create an Artifact.
        """
        self.path = f"{artifact_id}.{self.DEFAULT_EXT}"
        self.id = artifact_id
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        super().__init__()

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        """This function is called by the YAML loader to convert a YAML node
        into an Artifact object.

        It should not be called directly.
        """
        data = loader.construct_mapping(node, deep=True)
        return data["class"](artifact_id=data["artifact_id"])

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: Artifact):
        """This function is called by the YAML dumper to convert an Artifact
        object into a YAML node.

        It should not be called directly.
        """

        data = {
            "artifact_id": data.id,
            "class": type(data),
        }
        node = dumper.represent_mapping("!Artifact", data)
        return node

    def _ensure_downloaded(self):
        if not self.is_downloaded:
            raise ValueError(
                "Cannot load `Artifact` that has not been downloaded. "
                "First call `artifact.download()`."
            )


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

    def load(self) -> Any:
        self._ensure_downloaded()
        return yaml.load(open(self.local_path), Loader=yaml.FullLoader)

    def save(self, data: Any) -> None:
        return yaml.dump(data, open(self.local_path, "w"))


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
            dp = mk.datasets.get(name, dataset_dir=config.imagenet_dir, download=False)
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
            dp = mk.datasets.get(
                self.id, dataset_dir=config.imagenet_dir, download=False
            )
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
