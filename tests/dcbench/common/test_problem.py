import pytest

import dcbench


@pytest.fixture(params=["slice_discovery"])
def problem_class(request):
    task = request.param
    if task == "slice_discovery":
        return dcbench.slice_discovery
    else:
        raise ValueError(f"Task '{task}' not supported.")
