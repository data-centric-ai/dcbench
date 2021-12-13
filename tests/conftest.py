# contents of conftest.py
import os

import pytest


@pytest.fixture(autouse=True)
def set_test_bucket(monkeypatch):
    test_bucket_name = "dcbench"
    monkeypatch.setattr("dcbench.config.public_bucket_name", test_bucket_name)


@pytest.fixture(autouse=True)
def set_test_local(monkeypatch, tmpdir):
    monkeypatch.setattr("dcbench.config.local_dir", os.path.join(tmpdir, ".dcbench"))
