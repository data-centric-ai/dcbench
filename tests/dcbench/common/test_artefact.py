from dcbench.common.artefact import CSVArtefact


def test_remote_url():
    artefact = CSVArtefact("abc")
    assert (
        artefact.remote_url
        == "https://storage.googleapis.com/dcai/dcbench/artefacts/abc.csv"
    )
