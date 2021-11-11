import meerkat as mk
import numpy as np
import pandas as pd
import pytest

from dcbench.common.artifact import CSVArtifact, DataPanelArtifact


@pytest.fixture(params=["csv"])
def artifact(request):
    artifact_type = request.param

    artifact_id = f"test_{artifact_type}"
    if artifact_type == "csv":
        return CSVArtifact.from_data(
            pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}), artifact_id=artifact_id
        )
    elif artifact_type == "datapanel":
        return DataPanelArtifact.from_data(
            mk.DataPanel({"a": np.arange(5), "b": np.ones(5)}, artifact_id=artifact_id)
        )
    else:
        raise ValueError(f"Artifact type '{artifact_type}' not supported.")


@pytest.mark.skip(reason="requires gcloud authentication")
def test_artifact_upload(artifact):
    artifact.upload()
    # assert isinstance(artifact, CSVArtifact)


def test_artifact_download(artifact):
    artifact.download()
