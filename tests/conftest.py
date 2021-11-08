# contents of conftest.py
import pytest


@pytest.fixture(autouse=True)
def set_test_bucket(monkeypatch):
    test_bucket_name = "dcbench-test"
    monkeypatch.setattr("dcbench.constants.BUCKET_NAME", test_bucket_name)
    monkeypatch.setattr(
        "dcbench.constants.PUBLIC_REMOTE_URL",
        f"https://storage.googleapis.com/{test_bucket_name}",
    )


@pytest.fixture(autouse=True)
def set_test_local(monkeypatch, tmpdir):
    monkeypatch.setattr("dcbench.constants.LOCAL_DIR", tmpdir)
