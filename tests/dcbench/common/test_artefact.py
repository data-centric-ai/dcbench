import meerkat as mk
import numpy as np
import pandas as pd
import pytest

from dcbench.common.artefact import CSVArtefact, DataPanelArtefact


@pytest.fixture(params=["csv"])
def artefact(request):
    artefact_type = request.param

    artefact_id = f"test_{artefact_type}"
    if artefact_type == "csv":
        return CSVArtefact.from_data(
            pd.DataFrame({"a": np.arange(5), "b": np.ones(5)}), artefact_id=artefact_id
        )
    elif artefact_type == "datapanel":
        return DataPanelArtefact.from_data(
            mk.DataPanel({"a": np.arange(5), "b": np.ones(5)}, artefact_id=artefact_id)
        )
    else:
        raise ValueError(f"Artefact type '{artefact_type}' not supported.")


@pytest.mark.skip(reason="requires gcloud authentication")
def test_artefact_upload(artefact):
    artefact.upload()
    # assert isinstance(artefact, CSVArtefact)


def test_artefact_download(artefact):
    artefact.download()
