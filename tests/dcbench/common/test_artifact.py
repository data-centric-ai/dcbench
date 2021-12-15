import os
import shutil
from typing import Any

import meerkat as mk
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
import yaml

from dcbench.common.artifact import (
    Artifact,
    CSVArtifact,
    DataPanelArtifact,
    ModelArtifact,
    YAMLArtifact,
)
from dcbench.common.artifact_container import ArtifactSpec
from dcbench.common.modeling import Model


class SimpleModel(Model):
    def _set_model(self):
        self.layer = nn.Linear(in_features=self.config["in_features"], out_features=2)


@pytest.fixture(params=["csv", "datapanel", "model", "yaml"])
def artifact(request):
    artifact_type = request.param

    artifact_id = f"test_{artifact_type}"
    if artifact_type == "csv":
        return CSVArtifact.from_data(
            pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}), artifact_id=artifact_id
        )
    elif artifact_type == "datapanel":
        return DataPanelArtifact.from_data(
            mk.DataPanel({"a": np.arange(5), "b": np.ones(5)}), artifact_id=artifact_id
        )
    elif artifact_type == "model":
        return ModelArtifact.from_data(
            SimpleModel({"in_features": 4}), artifact_id=artifact_id
        )
    elif artifact_type == "yaml":
        return YAMLArtifact.from_data([1, 2, 3], artifact_id=artifact_id)
    else:
        raise ValueError(f"Artifact type '{artifact_type}' not supported.")


def is_data_equal(data1: Any, data2: Any) -> bool:
    if not isinstance(data1, type(data2)):
        return False

    if isinstance(data1, pd.DataFrame):
        return data1.equals(data2)
    elif isinstance(data1, mk.DataPanel):
        for col in data1.columns:
            if col not in data2.columns:
                return False
            if not (data1[col] == data2[col]).all():
                return False
    elif isinstance(data1, Model):
        return (
            data1.layer.weight == data2.layer.weight
        ).all() and data1.config == data2.config
    elif isinstance(data1, (list, dict)):
        return data1 == data2
    else:
        raise ValueError(f"Data type '{type(data1)}' not supported.")
    return True


def test_artifact_upload_download(set_tmp_bucket, artifact):
    data = artifact.load()
    uploaded = artifact.upload(force=True)
    assert uploaded
    downloaded = artifact.download(force=True)
    assert downloaded
    assert is_data_equal(data, artifact.load())

    # check that upload without force does not upload
    uploaded = artifact.upload()
    assert not uploaded

    # check that download without force does not download
    downloaded = artifact.download()
    assert not downloaded


def test_to_yaml_from_yaml(artifact):
    yaml_str = yaml.dump(artifact)
    artifact_from_yaml = yaml.load(yaml_str, Loader=yaml.FullLoader)
    assert artifact_from_yaml.id == artifact.id
    assert isinstance(artifact.load(), type(artifact_from_yaml.load()))
    assert artifact.remote_url == artifact_from_yaml.remote_url
    assert artifact.local_path == artifact.local_path
    assert is_data_equal(artifact.load(), artifact_from_yaml.load())


def test_load_without_download_errors(artifact):
    if os.path.isdir(artifact.local_path):
        shutil.rmtree(artifact.local_path)
    else:
        os.remove(artifact.local_path)

    with pytest.raises(ValueError) as excinfo:
        artifact.load()

    assert "`Artifact`" in str(excinfo.value)


def test_upload_without_save_errors(artifact):
    if os.path.isdir(artifact.local_path):
        shutil.rmtree(artifact.local_path)
    else:
        os.remove(artifact.local_path)

    with pytest.raises(ValueError) as excinfo:
        artifact.upload()

    assert "Artifact" in str(excinfo.value)


def test_from_data():
    artifact = Artifact.from_data(pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}))
    assert isinstance(artifact, CSVArtifact)

    artifact = Artifact.from_data(mk.DataPanel({"a": np.arange(5), "b": np.ones(5)}))
    assert isinstance(artifact, DataPanelArtifact)

    artifact = Artifact.from_data(SimpleModel({"in_features": 4}))
    assert isinstance(artifact, ModelArtifact)

    artifact = Artifact.from_data([1, 2, 3])
    assert isinstance(artifact, YAMLArtifact)

    with pytest.raises(ValueError) as excinfo:
        Artifact.from_data(None)
    assert "Artifact" in str(excinfo.value)
