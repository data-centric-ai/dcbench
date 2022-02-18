import meerkat as mk
import numpy as np
import pandas as pd
import pytest
import yaml

from dcbench.common.artifact import Artifact, CSVArtifact, DataPanelArtifact
from dcbench.common.artifact_container import ArtifactContainer, ArtifactSpec
from dcbench.common.table import AttributeSpec

from .test_artifact import is_data_equal


class SimpleContainer(ArtifactContainer):
    artifact_specs = {
        "csv1": ArtifactSpec("Description of csv artifact", CSVArtifact),
        "csv2": ArtifactSpec("Description of common csv artifact", CSVArtifact),
        "dp1": ArtifactSpec("Description of datapanel artifact", DataPanelArtifact),
        "dp2": ArtifactSpec(
            "Description of datapanel artifact", DataPanelArtifact, optional=True
        ),
    }
    attribute_specs = {
        "test_attr": AttributeSpec(
            description="A test attribute", attribute_type=int, optional=True
        )
    }
    task_id = "test_task"
    container_type = "test_container_type"


@pytest.fixture
def container():
    return SimpleContainer(
        artifacts={
            "csv1": CSVArtifact.from_data(
                pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}),
                artifact_id="csv1",
            ),
            "dp1": DataPanelArtifact.from_data(
                mk.DataPanel({"a": np.arange(5), "b": np.ones(5)}),
                artifact_id="dp1",
            ),
            "csv2": CSVArtifact.from_data(
                pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}),
                artifact_id="csv2",
            ),
        },
        attributes={
            "test_attr": 5,
        },
        container_id="test_container",
    )


def test_artifact_container_from_raw_data():

    df = pd.DataFrame({"a": np.arange(5), "b": np.ones(5)})
    dp = mk.DataPanel({"a": np.arange(5), "b": np.ones(5)})

    artifact = Artifact.from_data(df, artifact_id="common_artifact")

    container = SimpleContainer(
        artifacts={"csv1": df, "dp1": dp, "csv2": artifact},
        container_id="test_container",
    )

    assert is_data_equal(container.artifacts["csv1"].load(), df)
    assert is_data_equal(container.artifacts["dp1"].load(), dp)
    assert is_data_equal(container.artifacts["csv2"].load(), df)

    # if the container is passed raw object, an ID is automatically generated for it
    assert (
        container.artifacts["csv1"].id
        == "test_task/test_container_type/artifacts/test_container/csv1"
    )
    assert (
        container.artifacts["dp1"].id
        == "test_task/test_container_type/artifacts/test_container/dp1"
    )
    assert container.artifacts["csv2"].id == "common_artifact"


def test_artifact_container_invalid_artifact(container):
    df = pd.DataFrame({"a": np.arange(5), "b": np.ones(5)})
    dp = mk.DataPanel({"a": np.arange(5), "b": np.ones(5)})

    with pytest.raises(ValueError) as excinfo:
        SimpleContainer(
            artifacts={"csv1": df, "dp1": dp, "csv2": df, "nonexistent": df},
            container_id="test_container",
        )
    assert "Passed artifact name 'nonexistent'" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        SimpleContainer(
            artifacts={"csv1": df, "csv2": Artifact.from_data(dp), "dp1": dp},
            container_id="test_container",
        )
    assert "for the artifact named 'csv2'" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        SimpleContainer(
            artifacts={"dp1": dp, "csv2": df},
            container_id="test_container",
        )

    assert "Must pass required artifact with key 'csv1'" in str(excinfo.value)


@pytest.mark.parametrize("use_force", [True, False])
def test_artifact_container_download(monkeypatch, container, use_force: bool):
    downloads = []

    # mock the download function
    def mock_download(self, force: str = True):
        if not use_force:
            return False
        downloads.append(self.id)
        return True

    monkeypatch.setattr(Artifact, "download", mock_download)

    downloaded = container.download(force=use_force)
    assert downloaded == use_force

    if use_force:
        assert len(downloads) == 3
        assert set(downloads) == set(["csv1", "dp1", "csv2"])
    else:
        assert len(downloads) == 0


@pytest.mark.parametrize("use_force", [True, False])
def test_artifact_container_upload(monkeypatch, container, use_force: bool):
    uploads = []

    # mock the upload function
    def mock_upload(self, force: str = False, bucket: str = None):
        if not force:
            return False
        uploads.append(self.id)
        return True

    monkeypatch.setattr(Artifact, "upload", mock_upload)

    uploaded = container.upload(force=use_force)
    assert uploaded == use_force

    if use_force:
        assert len(uploads) == 3
        assert set(uploads) == set(["csv1", "dp1", "csv2"])
    else:
        assert len(uploads) == 0


def test_artifact_container_is_uploaded(monkeypatch, container):
    def mock_is_uploaded(self):
        return True

    monkeypatch.setattr(Artifact, "is_uploaded", property(mock_is_uploaded))

    is_uploaded = container.is_uploaded
    assert is_uploaded


def test_artifact_container_is_not_uploaded(monkeypatch, container):
    def mock_is_not_uploaded(self):
        print(self.id)
        return self.id != "csv1"

    monkeypatch.setattr(Artifact, "is_uploaded", property(mock_is_not_uploaded))

    is_uploaded = container.is_uploaded
    assert not is_uploaded


def test_artifact_container_is_downloaded(monkeypatch, container):
    def mock_is_downloaded(self):
        return True

    monkeypatch.setattr(Artifact, "is_downloaded", property(mock_is_downloaded))

    is_downloaded = container.is_downloaded
    assert is_downloaded


def test_artifact_container_is_not_downloaded(monkeypatch, container):
    def mock_is_downloaded(self):
        return self.id != "csv1"

    monkeypatch.setattr(Artifact, "is_downloaded", property(mock_is_downloaded))

    is_downloaded = container.is_downloaded
    assert not is_downloaded


def test_artifact_container_to_yaml_from_yaml(container):
    yaml_str = yaml.dump(container)
    container_from_yaml = yaml.load(yaml_str, Loader=yaml.FullLoader)

    assert container.id == container_from_yaml.id
    assert isinstance(container, type(container_from_yaml))
    assert is_data_equal(
        container.artifacts["csv1"].load(), container_from_yaml.artifacts["csv1"].load()
    )
    assert is_data_equal(
        container.artifacts["dp1"].load(), container_from_yaml.artifacts["dp1"].load()
    )
    assert is_data_equal(
        container.artifacts["csv2"].load(), container_from_yaml.artifacts["csv2"].load()
    )


def test_artifact_container_repr(container):
    assert "SimpleContainer" in str(container)


def test_artifact_container_len(container):
    assert len(container) == 3


def test_artifact_container_getitem(container):
    assert is_data_equal(container["csv1"], container.artifacts["csv1"].load())
    assert is_data_equal(container["dp1"], container.artifacts["dp1"].load())
    assert is_data_equal(container["csv2"], container.artifacts["csv2"].load())


def test_artifact_container_iter(container):

    assert [key for key in container] == ["csv1", "dp1", "csv2"]


def test_attribute_access(container):
    assert container.test_attr == 5

    with pytest.raises(AttributeError):
        container.nonexistent_attr
