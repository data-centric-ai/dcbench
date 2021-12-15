from __future__ import annotations

import os
import uuid
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union

import yaml
from meerkat.tools.lazy_loader import LazyLoader

import dcbench.constants as constants
from dcbench.config import config

from .table import AttributeSpec, RowMixin

storage = LazyLoader("google.cloud.storage")


from .artifact import Artifact

PRIMITIVE_TYPE = Union[int, float, str, bool]


@dataclass
class ArtifactSpec:
    description: str
    artifact_type: type
    optional: bool = False


class ArtifactContainer(ABC, Mapping, RowMixin):
    """A logical collection of artifacts and attributes (simple tags describing the
    container), which are useful for finding, sorting and grouping containers.

    Args:
        artifacts (Mapping[str, Union[Artifact, Any]]): A mapping with the same keys
            as the `ArtifactContainer.artifact_specs` (possibly excluding optional
            artifacts). Each value can either be an :class:`Artifact`, in which case the
            artifact type must match the type specified in the corresponding
            :class:`ArtifactSpec`, or a raw object, in which case a new artifact of the
            type specified in `artifact_specs` is created from the raw object and an
            ``artifact_id`` is generated according to the following pattern:
            ``<task_id>/<container_type>/artifacts/<container_id>/<key>``.
        attributes (Mapping[str, PRIMITIVE_TYPE], optional): A mapping with the same
            keys as the `ArtifactContainer.attribute_specs` (possibly excluding optional
            attributes). Each value must be of the type specified in the corresponding
            :class:`AttributeSpec`. Defaults to None.
        container_id (str, optional): The ID of the container. Defaults to None, in
            which case a UUID is generated.

    Notes
    -----

    :class:`ArtifactContainer` is an abstract base class, and should not be
    instantiated directly. There are two main groups of :class:`ArtifactContainer`
    subclasses:

    #. :class:`dcbench.Problem` - A logical collection of artifacts and
       attributes that correspond to a specific problem to be solved.

       - Example subclasses: :class:`dcbench.SliceDiscoveryProblem`,
         :class:`dcbench.BudgetcleanProblem`
    #. :class:`dcbench.Solution` - A logical collection of artifacts and
       attributes that correspond to a solution to a problem.

       - Example subclasses: :class:`dcbench.SliceDiscoverySolution`,
         :class:`dcbench.BudgetcleanSolution`

    A concrete (i.e. non-abstract) subclass of :class:`ArtifactContainer` must include
    (1) a specification for the artifacts it holds, (2) a specification for the
    attributes used to tag it, and (3) a `task_id` linking the subclass
    to one of dcbench's tasks (see :ref:`task-intro`). For example, in the code block
    below we include such a specification in the definition of a simple container that
    holds a training dataset and a test dataset (see
    :class:`dcbench.SliceDiscoveryProblem` for a real example):

    .. code-block:: python

        class DemoContainer(ArtifactContainer):
            artifact_specs = {
                "train_dataset": ArtifactSpec(
                    artifact_type=CSVArtifact,
                    description="A CSV containing training data."
                ),
                "test_dataset": ArtifactSpec(
                    artifact_type=CSVArtifact,
                    description="A CSV containing test data."
                ),
            }
            attribute_specs = {
                "dataset_name": AttributeSpec(
                    attribute_type=str,
                    description="The name of the dataset."
                ),
            }
            task_id = "slice_discovery"

    """

    artifact_specs: Mapping[str, ArtifactSpec]
    task_id: str
    attribute_specs: Mapping[str, AttributeSpec] = {}

    # abstract subclasses like Problem and Solution specify this so that all of their
    # subclasses may be grouped by container_type when stored on disk
    container_type: str = "artifact_container"

    def __init__(
        self,
        artifacts: Mapping[str, Artifact],
        attributes: Mapping[str, PRIMITIVE_TYPE] = None,
        container_id: str = None,
    ):
        if container_id is None:
            container_id = uuid.uuid4().hex

        super().__init__(id=container_id)
        self._check_artifact_specs(artifacts=artifacts)
        artifacts = self._create_artifacts(artifacts=artifacts)
        self.artifacts = artifacts

        if attributes is None:
            attributes = {}
        self.attributes = attributes  # This setter will check the artifact_specs

    @property
    def is_downloaded(self) -> bool:
        """Checks if all of the artifacts in the container are downloaded to the local
        directory specified in the config file at ``config.local_dir``.

        Returns:
            bool: True if artifact is downloaded, False otherwise.
        """
        return all(x.is_downloaded for x in self.artifacts.values())

    @property
    def is_uploaded(self) -> bool:
        """Checks if all of the artifacts in the container are uploaded to the GCS
        bucket specified in the config file at ``config.public_bucket_name``.

        Returns:
            bool: True if artifact is uploaded, False otherwise.
        """
        return all(x.is_uploaded for x in self.artifacts.values())

    def upload(self, force: bool = False, bucket: "storage.Bucket" = None):
        """Uploads all of the artifacts in the container to a GCS bucket, skipping
        artifacts that are already uploaded.

        Args:
            force (bool, optional): Force upload even if an artifact is already
                uploaded. Defaults to False.
            bucket (storage.Bucket, optional): The GCS bucket to which the artifacts are
                uploaded. Defaults to None, in which case the artifact is uploaded to
                the bucket speciried in the config file at config.public_bucket_name.

        Returns:
            bool: True if any artifacts were uploaded, False otherwise.
        """
        if bucket is None:
            client = storage.Client()
            bucket = client.get_bucket(config.public_bucket_name)

        return any(
            [
                artifact.upload(force=force, bucket=bucket)
                for artifact in self.artifacts.values()
            ]
        )

    def download(self, force: bool = False) -> bool:
        """Downloads artifacts in the container from the GCS bucket specified in the
        config file at ``config.public_bucket_name`` to the local directory specified
        in the config file at ``config.local_dir``. The relative path to the
        artifact within that directory is ``self.path``, which by default is
        just the artifact ID with the default extension.

        Args:
            force (bool, optional): Force download even if an artifact is already
             downloaded. Defaults to False.

        Returns:
            bool: True if any artifacts were downloaded, False otherwise.
        """
        return any(
            [artifact.download(force=force) for artifact in self.artifacts.values()]
        )

    @staticmethod
    def from_yaml(loader: yaml.Loader, node):
        """This function is called by the YAML loader to convert a YAML node
        into an :class:`ArtifactContainer` object.

        It should not be called directly.
        """
        data = loader.construct_mapping(node, deep=True)
        return data["class"](
            container_id=data["container_id"],
            artifacts=data["artifacts"],
            attributes=data["attributes"],
        )

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: ArtifactContainer):
        """This function is called by the YAML dumper to convert an
        :class:`ArtifactContainer` object into a YAML node.

        It should not be called directly.
        """
        data = {
            "class": type(data),
            "container_id": data.id,
            "attributes": data._attributes,
            "artifacts": data.artifacts,
        }
        return dumper.represent_mapping("!ArtifactContainer", data)

    # Provide dict interface for accessing artifacts by name
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

    def __repr__(self):
        artifacts = {k: v.__class__.__name__ for k, v in self.artifacts.items()}
        return (
            f"{self.__class__.__name__}(artifacts={artifacts}, "
            f"attributes={self.attributes})"
        )

    @classmethod
    def _check_artifact_specs(cls, artifacts: Mapping[str, Artifact]):
        for name, artifact in artifacts.items():
            if name not in cls.artifact_specs:
                raise ValueError(
                    f"Passed artifact name '{name}', but the specification for"
                    f" {cls.__name__} doesn't include it."
                )

            # defer the check to see if an artifact can actually be created from the raw
            # data to _create_artifacts
            if isinstance(artifact, Artifact) and not isinstance(
                artifact, cls.artifact_specs[name].artifact_type
            ):
                raise ValueError(
                    f"Passed an artifact of type {type(artifact)} to {cls.__name__}"
                    f" for the artifact named '{name}'. The specification for"
                    f" {cls.__name__} expects an Artifact of type"
                    f" {cls.artifact_specs[name].artifact_type}."
                )

        for name, spec in cls.artifact_specs.items():
            if name not in artifacts:
                if spec.optional:
                    continue
                raise ValueError(
                    f"Must pass required artifact with key '{name}' to {cls.__name__}."
                )

    def _create_artifacts(self, artifacts: Mapping[str, Artifact]):
        return {
            name: artifact
            if isinstance(artifact, Artifact)
            else self.artifact_specs[name].artifact_type.from_data(
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


yaml.add_multi_representer(ArtifactContainer, ArtifactContainer.to_yaml)
yaml.add_constructor("!ArtifactContainer", ArtifactContainer.from_yaml)
