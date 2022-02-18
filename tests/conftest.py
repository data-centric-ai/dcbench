# contents of conftest.py
import os

import google.cloud.storage as storage
import pytest


@pytest.fixture(autouse=True)
def set_test_bucket(monkeypatch):
    test_bucket_name = "dcbench"
    monkeypatch.setattr("dcbench.config.public_bucket_name", test_bucket_name)


@pytest.fixture()
def set_tmp_bucket(monkeypatch):
    test_bucket_name = "dcbench-test"
    monkeypatch.setattr("dcbench.config.public_bucket_name", test_bucket_name)

    # code above this yield will be executed before every test
    yield
    # code below this yield will be executed after every test

    assert test_bucket_name != "dcbench"  # ensure we don't empty the production bucket
    client = storage.Client()
    bucket = client.get_bucket(test_bucket_name)
    blobs = list(bucket.list_blobs())
    bucket.delete_blobs(blobs)
    return test_bucket_name


@pytest.fixture(autouse=True)
def set_test_local(monkeypatch, tmpdir):
    monkeypatch.setattr("dcbench.config.local_dir", os.path.join(tmpdir, ".dcbench"))
